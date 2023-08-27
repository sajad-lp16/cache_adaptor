import copy
import time
import random
import logging
from copy import deepcopy
from functools import wraps
from typing import Callable
from urllib.parse import urlparse

from django.db.models import F
from django.urls import resolve
from django.conf import settings
from django.utils import timezone
from django.shortcuts import get_object_or_404
from rest_framework import status

from main_app.models import (
    BaseQuiz,
    UserQuiz,
    Question,
    BaseScoredQuiz
)
from cachelib.redis_gateway import RedisGateway
from utils.exceptions import (
    QuizException,
    QuestionRetrieveDenied,
)
from utils import exceptions

redis_logger = logging.getLogger("redis")

NO_DURATION = NO_MAX_PARTICIPATION = -1


class CounterCacheManager:
    _gateway = RedisGateway("COUNTER")

    user_participation_count = "user_id_{user_id}_quiz_id_{quiz_id}"
    quiz_max_participation_cache_key = "quiz_id_{quiz_id}_max_participation"

    @classmethod
    def get_quiz_max_participation(cls, quiz_id):
        key = cls.quiz_max_participation_cache_key.format(quiz_id=quiz_id)
        if (qua := cls._gateway.get(key, cast=int)) is not None:
            return qua
        quiz_instance = get_object_or_404(BaseQuiz.active_objects, id=quiz_id)
        return cls.set_quiz_max_participation(quiz_instance)

    @classmethod
    def set_quiz_max_participation(cls, quiz_instance):
        key = cls.quiz_max_participation_cache_key.format(quiz_id=quiz_instance.id)
        value = quiz_instance.max_participation
        if value is None:
            return
        cls._gateway.set(key, value, timeout=settings.COUNTER_CACHE_TIMEOUT)
        return value

    @classmethod
    def get_user_participation_count(cls, user_id, quiz_id):
        key = cls.user_participation_count.format(user_id=user_id, quiz_id=quiz_id)
        if (qua := cls._gateway.get(key, cast=int)) is not None:
            return qua
        return cls.set_user_participation_count(user_id, quiz_id)

    @classmethod
    def set_user_participation_count(cls, user_id, quiz_id):
        key = cls.user_participation_count.format(user_id=user_id, quiz_id=quiz_id)
        value = UserQuiz.active_objects.filter(user_id=user_id, quiz_id=quiz_id).count()
        cls._gateway.set(key, value, timeout=settings.COUNTER_CACHE_TIMEOUT)
        return value

    @classmethod
    def incr_user_participation_count(cls, user_id, quiz_id):
        if cls.get_quiz_max_participation(quiz_id) is None:
            return
        key = cls.user_participation_count.format(user_id=user_id, quiz_id=quiz_id)
        cls._gateway.incr(key)

    @classmethod
    def has_participation_permission(cls, user_id, quiz_id):
        if (quiz_mx := cls.get_quiz_max_participation(quiz_id)) is None:
            return True
        user_participation_count = cls.get_user_participation_count(user_id, quiz_id)
        if user_participation_count < quiz_mx:
            return True
        return False


class DetailCacheManager:
    _gateway = RedisGateway("QUIZ_DETAIL")

    quiz_detail_cache_key = "quiz_id_{quiz_id}_detail"

    @classmethod
    def get_quiz_detail(cls, quiz_id):
        key = cls.quiz_detail_cache_key.format(quiz_id=quiz_id)
        return cls._gateway.xget(key)

    @classmethod
    def delete_quiz_detail(cls, quiz_id):
        key = cls.quiz_detail_cache_key.format(quiz_id=quiz_id)
        cls._gateway.delete_key(key)

    @classmethod
    def set_quiz_detail(cls, quiz_id, quiz_data):
        key = cls.quiz_detail_cache_key.format(quiz_id=quiz_id)
        cls._gateway.xset(key, quiz_data)


class StateCacheManager:
    _gateway = RedisGateway("QUIZ_STATE")

    _counter_cache = CounterCacheManager

    _USER_QUIZ_STATE_SCHEMA = {
        "user_id": None,
        "quiz_id": None,
        "show_answer_after_question": False,
        "show_answer_sheet_button": False,
        "quiz_duration": -1,
        "quiz_finish_timestamp": None,
        "question_links": [],
        "questions_state": [],
        "continue_ability": False,
        "back_ability": False,
        "questions_count": 0,
        "answers": {}
    }

    _user_quiz_state_cache_key = "user_id_{user_id}_quiz_id_{quiz_id}"
    _required_question_cache_key = "required_{question_id}"

    class Constraints:
        @staticmethod
        def check_for_state(func: Callable) -> Callable:
            @wraps(func)
            def inner(klass, request, user_quiz_state, *args, **kwargs):
                if user_quiz_state is not None:
                    return func(klass, request, user_quiz_state, *args, **kwargs)
                raise QuizException(
                    error_code=exceptions.QUIZ_EXPIRED_OR_NOT_PARTICIPATE_ERROR_CODE,
                )

            return inner

        @staticmethod
        def check_quiz_timeout(func):
            @wraps(func)
            def inner(klass, request, user_quiz_state, question_id):
                if StateCacheManager.quiz_is_expired(user_quiz_state):
                    StateCacheManager.state_2_database(request, user_quiz_state.get("quiz_id"))
                    raise QuizException(
                        error_code=exceptions.QUIZ_EXPIRED_OR_NOT_PARTICIPATE_ERROR_CODE
                    )
                return func(klass, request, user_quiz_state, question_id)

            return inner

    @classmethod
    def _get_schema(cls):
        state = deepcopy(cls._USER_QUIZ_STATE_SCHEMA)
        return state

    @classmethod
    def get_user_quiz_state(cls, user_id: int, quiz_id: int) -> None | dict:
        key = cls._user_quiz_state_cache_key.format(user_id=user_id, quiz_id=quiz_id)
        if (user_quiz_state := cls._gateway.xget(key)) is None:
            return
        return user_quiz_state

    @classmethod
    def _set_user_quiz_state(cls, user_id: int, quiz_id: int, user_quiz_state: dict) -> None:
        key = cls._user_quiz_state_cache_key.format(user_id=user_id, quiz_id=quiz_id)
        cls._gateway.xset(key, user_quiz_state)

    @staticmethod
    def build_question_link(request, question: int | Question) -> str:
        if isinstance(question, int):
            question = get_object_or_404(Question.active_objects, id=question)
        return request.build_absolute_uri(question.get_absolute_url())

    @classmethod
    def get_question_id_from_url(cls, question_link: str) -> int:
        url_path = urlparse(question_link).path
        return resolve(url_path).captured_kwargs.get("question_id")

    @classmethod
    def remove_all_related_states(cls, quiz_id: int) -> None:
        key = cls._user_quiz_state_cache_key.format(user_id="*", quiz_id=quiz_id)
        cls._gateway.remove_by_pattern(key)

    @classmethod
    def can_continue(cls, user_quiz_state) -> bool:
        if user_quiz_state is None:
            return False
        return user_quiz_state["continue_ability"]

    @classmethod
    def can_go_back(cls, user_quiz_state: dict) -> bool:
        if user_quiz_state is None:
            return False
        return user_quiz_state["back_ability"]

    @classmethod
    def get_quiz_remaining_time(cls, user_quiz_state: dict) -> None | int:
        if user_quiz_state.get("quiz_duration") == NO_DURATION:
            return NO_DURATION
        return user_quiz_state["quiz_finish_timestamp"] - int(time.time())

    @staticmethod
    def get_questions_state(user_quiz_state):
        return {
            "previous_question_link": user_quiz_state["questions_state"][0],
            "current_question_link": user_quiz_state["questions_state"][1],
            "next_question_link": user_quiz_state["questions_state"][2],
        }

    @classmethod
    def quiz_is_expired(cls, user_quiz_state: dict) -> bool:
        remaining_time = cls.get_quiz_remaining_time(user_quiz_state)
        if remaining_time == NO_DURATION:
            return False
        return remaining_time + settings.LATENCY_TIME < 0

    @classmethod
    def remove_state(cls, user_id: int, quiz_id: int):
        key = cls._user_quiz_state_cache_key.format(user_id=user_id, quiz_id=quiz_id)
        cls._gateway.delete_key(key)

    @classmethod
    def perform_create_answer_models(cls, user_quiz_state: dict, user_quiz: UserQuiz) -> int:
        check_required_count = 0
        all_submitted_answers_dict = user_quiz_state.get("answers")
        for question_id, answer_data in all_submitted_answers_dict.items():
            answer_type, data, grade, creation_date = answer_data
            answer_model = Question.get_answer_model_class(answer_type)
            check_required = answer_model.perform_create_model(user_quiz, question_id, data, grade, creation_date)
            check_required_count += bool(check_required)
        return check_required_count

    @classmethod
    def _state_2_database(cls, user_quiz_state):
        if user_quiz_state is None:
            raise QuizException(
                error_code=exceptions.QUIZ_EXPIRED_OR_NOT_PARTICIPATE_ERROR_CODE,
            )
        user_id = user_quiz_state.get("user_id")
        quiz_id = user_quiz_state.get("quiz_id")
        quiz_instance = get_object_or_404(BaseQuiz, id=quiz_id)
        related_quiz_instance = quiz_instance.related_model
        user_quiz = UserQuiz.active_objects.filter(
            user_id=user_id,
            quiz=related_quiz_instance,
            finish_time__isnull=True
        ).last()
        if user_quiz is None:
            cls.remove_state(user_id, quiz_id)
            raise QuizException(
                error_code=exceptions.QUIZ_EXPIRED_OR_NOT_PARTICIPATE_ERROR_CODE,
            )
        user_quiz.finish_time = timezone.now()
        check_required_count = cls.perform_create_answer_models(user_quiz_state, user_quiz)
        questions_count = user_quiz_state.get("questions_count")
        if isinstance(related_quiz_instance, BaseScoredQuiz):
            corrects = user_quiz.correct_answers_count
            wrongs = user_quiz.wrong_answers_count
            if (total_answered := wrongs + corrects + check_required_count) != questions_count:
                user_quiz.wrong_answers_count += questions_count - total_answered
        user_quiz.save()
        return user_quiz, user_quiz_state

    @classmethod
    def state_2_database(cls, request, quiz_id: int):
        if (user_quiz_state := cls.get_user_quiz_state(request.user_id, quiz_id)) is not None:
            return cls._state_2_database(user_quiz_state)
        return None, None

    @classmethod
    def check_expired_states(cls):
        all_states_keys = cls._gateway.get_pattern("*")
        for key in all_states_keys:
            state = cls._gateway.xget(key, combine=False)
            if state.get("quiz_duration") != NO_DURATION:
                if not cls.quiz_is_expired(state):
                    continue
                cls._state_2_database(state)


class StartFinishStateCache(StateCacheManager):
    class Constraints(StateCacheManager.Constraints):
        @staticmethod
        def check_max_participation(func) -> Callable:
            @wraps(func)
            def inner(klass, request, user_id, quiz_id):
                if CounterCacheManager.has_participation_permission(user_id, quiz_id):
                    return func(klass, request, user_id, quiz_id)
                raise QuizException(
                    error_code=exceptions.MAXIMUM_PARTICIPATION_ERROR_CODE
                )

            return inner

        @staticmethod
        def check_continue_ability(func) -> Callable:
            @wraps(func)
            def inner(klass, request, user_quiz_state):
                if not StartFinishStateCache.can_continue(user_quiz_state):
                    StartFinishStateCache.state_2_database(request, user_quiz_state.get("quiz_id"))
                    raise QuizException(
                        error_code=exceptions.CONTINUE_ABILITY_ERROR_CODE
                    )
                return func(klass, request, user_quiz_state)

            return inner

    @classmethod
    def _get_finish_response_schema(cls):
        return copy.deepcopy({
            "start_time": "",
            "finish_time": "",
            "grade": None,
            "total_grade": None,
            "total_questions": None,
            "correct_answers_count": None,
            "wrong_answers_count": None,
            "show_answer_sheet_button": None,
            "answer_stats": []
        })

    @classmethod
    def _get_participation_response(cls, user_quiz_state: dict) -> dict:
        response = {
            **cls.get_questions_state(user_quiz_state),
            "remaining_duration": cls.get_quiz_remaining_time(user_quiz_state)
        }
        response["next_question_url"] = response.pop("current_question_link")
        response.pop("next_question_link")
        response["purchased"] = 1
        return response

    @classmethod
    def _create_user_quiz_schema(cls, request, quiz_instance: BaseQuiz) -> dict:
        user_quiz_state = cls._get_schema()
        questions = quiz_instance.questions.order_by(
            "id"
        ).all()

        user_quiz_state["user_id"] = request.user_id
        user_quiz_state["quiz_id"] = quiz_instance.id
        user_quiz_state["continue_ability"] = quiz_instance.continue_ability
        user_quiz_state["back_ability"] = quiz_instance.back_ability
        user_quiz_state["quiz_duration"] = quiz_instance.duration or NO_DURATION
        if quiz_instance.duration is not None:
            user_quiz_state["quiz_finish_timestamp"] = quiz_instance.duration + int(time.time())

        for question in questions:
            if question.is_required:
                user_quiz_state[cls._required_question_cache_key.format(question_id=question.id)] = 1
        question_links = [cls.build_question_link(request, question) for question in questions]
        if quiz_instance.randomize_questions:
            random.shuffle(question_links)
        user_quiz_state["question_links"] = question_links
        user_quiz_state["questions_count"] = len(questions)
        user_quiz_state["questions_state"] = [None, *question_links[:2], None][:3]
        if isinstance(q_model := quiz_instance.related_model, BaseScoredQuiz):
            user_quiz_state["show_answer_after_question"] = q_model.show_answer_after_question
            user_quiz_state["show_answer_sheet_button"] = q_model.show_answer_sheet_button
        return user_quiz_state

    @classmethod
    @Constraints.check_max_participation
    def perform_new_user_quiz_cache(cls, request, user_id: int, quiz_id: int) -> dict:
        cls.state_2_database(request, quiz_id)
        quiz_instance = get_object_or_404(BaseQuiz.active_objects, id=quiz_id)
        user_quiz_state = cls._create_user_quiz_schema(request=request, quiz_instance=quiz_instance)
        related_quiz_instance = quiz_instance.related_model
        user_quiz = UserQuiz(
            user_id=request.user_id,
            quiz=related_quiz_instance,
            questions_count=user_quiz_state.get("questions_count"),
        )
        if isinstance(related_quiz_instance, BaseScoredQuiz):
            user_quiz.wrong_answers_count = 0
            user_quiz.correct_answers_count = 0
            user_quiz.grade = 0

        user_quiz.save()
        CounterCacheManager.incr_user_participation_count(request.user_id, quiz_id)
        cls._set_user_quiz_state(user_id, quiz_id, user_quiz_state)

        return cls._get_participation_response(user_quiz_state)

    @classmethod
    @Constraints.check_continue_ability
    def perform_continue_quiz(cls, request, user_quiz_state: dict) -> dict:
        return cls._get_participation_response(user_quiz_state)

    @classmethod
    def perform_user_quiz_cache(cls, request, user_id, quiz_id):
        if (user_quiz_state := cls.get_user_quiz_state(user_id, quiz_id)) is not None:
            return cls.perform_continue_quiz(request, user_quiz_state), status.HTTP_200_OK
        return cls.perform_new_user_quiz_cache(request, user_id, quiz_id), status.HTTP_201_CREATED

    @classmethod
    def perform_finish_quiz(cls, request, quiz_id):
        user_quiz, user_quiz_state = cls.state_2_database(request, quiz_id)
        if user_quiz is None:
            raise QuizException(
                error_code=exceptions.QUIZ_EXPIRED_OR_NOT_PARTICIPATE_ERROR_CODE,
            )
        quiz = user_quiz.quiz.related_model

        finish_response_schema = cls._get_finish_response_schema()

        finish_response_schema["start_time"] = user_quiz.creation_date
        finish_response_schema["finish_time"] = user_quiz.finish_time
        finish_response_schema["grade"] = user_quiz.grade
        finish_response_schema["total_grade"] = getattr(quiz, "grade", None)
        finish_response_schema["total_questions"] = user_quiz.questions_count
        finish_response_schema["correct_answers_count"] = getattr(user_quiz, "correct_answers_count", None)
        finish_response_schema["wrong_answers_count"] = getattr(user_quiz, "wrong_answers_count", None)
        finish_response_schema["show_answer_sheet_button"] = getattr(quiz, "show_answer_sheet_button", None)

        answer_types = [item[0] for item in user_quiz_state["answers"].values()]

        for answer_type in set(answer_types):
            finish_response_schema["answer_stats"].append({
                "answer_type": answer_type,
                "answer_type_display": dict(Question.AnswerType.choices).get(answer_type),
                "total_questions_count": user_quiz.quiz.questions.filter(answer_type=answer_type).count(),
                "total_questions_user_answered": user_quiz.user_answers.filter(
                    question__answer_type=answer_type
                ).count(),
                "total_questions_user_answered_correctly": user_quiz.user_answers.filter(
                    question__answer_type=answer_type, grade=F("question__grade")
                ).count()
            })
        cls.remove_state(request.user_id, quiz_id)
        return finish_response_schema


class HistoryCache(StateCacheManager):
    class Constraints(StateCacheManager.Constraints):
        pass

    @classmethod
    def get_on_going_quiz_cache_data(cls, user_id, quiz_id):
        user_quiz_state = cls.get_user_quiz_state(user_id, quiz_id)
        if user_quiz_state is None:
            raise QuizException(
                error_code=exceptions.QUIZ_EXPIRED_OR_NOT_PARTICIPATE_ERROR_CODE,
            )
        return {
            "quiz_max_participation": CounterCacheManager.get_quiz_max_participation(quiz_id),
            "user_participation_amount": UserQuiz.objects.filter(user_id=user_id, quiz_id=quiz_id).count(),
            "total_quiz_questions": user_quiz_state["questions_count"],
            "total_quiz_questions_answered": len(user_quiz_state["answers"].keys())
        }


class QuestionRetrieveCache(StateCacheManager):
    _finish_timestamp_cache_key = "question_{question_id}_finish_timestamp"

    PREVIOUS_QUESTION = 0
    CURRENT_QUESTION = 1
    NEXT_QUESTION = 2

    class Constraints(StateCacheManager.Constraints):

        @staticmethod
        def check_question_timeout(func):
            @wraps(func)
            def inner(klass, request, user_quiz_state, question_id):
                if not QuestionRetrieveCache.question_is_expired(user_quiz_state, question_id):
                    return func(klass, request, user_quiz_state, question_id)
                question_order = QuestionRetrieveCache.get_which_question(request, user_quiz_state, question_id)
                if not question_order == QuestionRetrieveCache.CURRENT_QUESTION:
                    user_quiz_state = QuestionRetrieveCache.perform_update_state(request, user_quiz_state, question_id)
                raise QuestionRetrieveDenied(
                    error_code=exceptions.QUESTION_EXPIRED_ERROR_CODE,
                    context=QuestionRetrieveCache.get_questions_state(user_quiz_state)
                )

            return inner

        @staticmethod
        def _check_next_question_access(request, user_quiz_state: dict) -> None:
            current_question_link = user_quiz_state["questions_state"][1]
            current_question_id = StateCacheManager.get_question_id_from_url(current_question_link)
            current_is_required = QuestionRetrieveCache.is_required_question(user_quiz_state, current_question_id)
            current_is_answered = QuestionRetrieveCache.is_answered_question(user_quiz_state, current_question_id)
            current_is_expired = QuestionRetrieveCache.question_is_expired(user_quiz_state, current_question_id)
            if not current_is_required:
                return
            if current_is_answered:
                return
            if current_is_expired:
                return
            raise QuestionRetrieveDenied(
                error_code=exceptions.NEXT_QUESTION_NO_ACCESS_ERROR_CODE,
                context=StateCacheManager.get_questions_state(user_quiz_state)
            )

        @staticmethod
        def _check_previous_question_access(request, user_quiz_state: dict, question_id) -> None:
            def _raise():
                raise QuestionRetrieveDenied(
                    error_code=exceptions.PREVIOUS_NO_ACCESS_ERROR_CODE,
                    context=QuestionRetrieveCache.get_questions_state(user_quiz_state)
                )

            current_question_link = user_quiz_state["question_links"][1]
            current_question_id = StateCacheManager.get_question_id_from_url(current_question_link)
            back_ability = StateCacheManager.can_go_back(user_quiz_state)

            current_is_required = QuestionRetrieveCache.is_required_question(user_quiz_state, current_question_id)
            current_is_answered = QuestionRetrieveCache.is_answered_question(user_quiz_state, current_question_id)
            current_is_expired = QuestionRetrieveCache.question_is_expired(user_quiz_state, current_question_id)

            if not back_ability:
                _raise()
            if not current_is_required:
                return
            if current_is_answered:
                return
            if current_is_expired:
                return
            _raise()

        @classmethod
        def check_question_access(cls, func):
            @wraps(func)
            def inner(klass, request, user_quiz_state, question_id):
                question_order = QuestionRetrieveCache.get_which_question(request, user_quiz_state, question_id)
                if question_order == QuestionRetrieveCache.NEXT_QUESTION:
                    cls._check_next_question_access(request, user_quiz_state)
                elif question_order == QuestionRetrieveCache.PREVIOUS_QUESTION:
                    cls._check_previous_question_access(request, user_quiz_state, question_id)
                return func(klass, request, user_quiz_state, question_id)

            return inner

    @classmethod
    def get_question_remaining_time(cls, user_quiz_state: dict, question_id: int) -> None | int:
        question_timeout = user_quiz_state.get(cls._finish_timestamp_cache_key.format(question_id=question_id))
        if question_timeout in [NO_DURATION, None]:
            return NO_DURATION
        return question_timeout - int(time.time())

    @classmethod
    def question_is_expired(cls, user_quiz_state: dict, question_id) -> bool:
        remaining_time = cls.get_question_remaining_time(user_quiz_state, question_id)
        if remaining_time == NO_DURATION:
            return False
        return remaining_time + settings.LATENCY_TIME < 0

    @classmethod
    def is_required_question(cls, user_quiz_state: dict, question_id: int) -> bool:
        return bool(user_quiz_state.get(cls._required_question_cache_key.format(question_id=question_id)))

    @classmethod
    def get_which_question(cls, request, user_quiz_state: dict, question_id: int, between_all=False) -> int:
        """
        @param request:
        @param user_quiz_state:
        @param question_id:
        @param between_all: if set this flag to True function will return the index
                            of the question among all questions
        @return: index of question between
                 1 : [previous, current, next] questions,
                 2 : all questions
        """
        if between_all:
            questions_state = user_quiz_state["question_links"]
        else:
            questions_state = user_quiz_state["questions_state"]
        question_link = cls.build_question_link(request, question_id)
        if question_link not in questions_state:
            raise QuestionRetrieveDenied(
                error_code=exceptions.QUESTION_NO_ACCESS,
                context=cls.get_questions_state(user_quiz_state)
            )
        return questions_state.index(question_link)

    @classmethod
    def is_answered_question(cls, user_quiz_state: dict, question_id: int) -> bool:
        return bool(
            user_quiz_state["answers"].get(question_id)
        )

    @classmethod
    def perform_update_state(cls, request, user_quiz_state: dict, question_instance: Question | int) -> dict:
        def _get_element(collection, _index):
            try:
                assert _index >= 0
                return collection[_index]
            except (IndexError, AssertionError):
                return None

        question_link = cls.build_question_link(request, question_instance)
        all_question_links = user_quiz_state.get("question_links")
        questions_state = user_quiz_state["questions_state"]
        shift = 1
        try:
            question_index_all = all_question_links.index(question_link)
            question_index = questions_state.index(question_link)
        except ValueError as err:
            redis_logger.error(
                f"error while getting index of {question_link} "
                f"from {all_question_links}"
                f"err = {err}"
            )
            raise

        if question_index == cls.CURRENT_QUESTION:
            return user_quiz_state

        quiz_id = user_quiz_state.get("quiz_id")
        while True:
            try:
                next_url = all_question_links[question_index_all + shift]
                next_q_id = QuestionRetrieveCache.get_question_id_from_url(next_url)
                if QuestionRetrieveCache.question_is_expired(user_quiz_state, next_q_id):
                    shift += 1
                    continue
                break
            except IndexError:
                break

        new_questions = [None]
        new_questions += [
            _get_element(
                all_question_links, index
            ) for index in [question_index_all + shift - 1, question_index_all + shift]
        ]
        if user_quiz_state.get("back_ability"):
            new_questions[0] = _get_element(all_question_links, question_index_all + shift - 2)

        user_quiz_state["questions_state"] = new_questions
        cls._set_user_quiz_state(request.user_id, quiz_id, user_quiz_state)
        return user_quiz_state

    @classmethod
    def _set_question_duration(cls, user_quiz_state: dict, question_instance: Question) -> dict:
        key = cls._finish_timestamp_cache_key.format(question_id=question_instance.id)
        if user_quiz_state.get(key) is None:
            if question_instance.duration is None:
                user_quiz_state[key] = NO_DURATION
            else:
                user_quiz_state[key] = question_instance.duration + int(time.time())
        return user_quiz_state

    @classmethod
    @Constraints.check_for_state
    @Constraints.check_quiz_timeout
    @Constraints.check_question_timeout
    @Constraints.check_question_access
    def retrieve_question_instance(cls, request, user_quiz_state: dict, question_id: int) -> Question:
        return get_object_or_404(Question.active_objects, id=question_id)

    @classmethod
    def retrieve_question_data(
            cls, request, user_id: int, quiz_id: int, question_id: int
    ) -> tuple[Question, dict]:
        user_quiz_state = StateCacheManager.get_user_quiz_state(user_id, quiz_id)
        question_instance = cls.retrieve_question_instance(request, user_quiz_state, question_id)
        new_quiz_state = cls._set_question_duration(user_quiz_state, question_instance)
        cls.perform_update_state(request, new_quiz_state, question_instance)
        return (
            question_instance,
            {
                "question_remain_duration": cls.get_question_remaining_time(user_quiz_state, question_id),
                "quiz_remain_duration": cls.get_quiz_remaining_time(user_quiz_state),
                "question_number": cls.get_which_question(request, user_quiz_state, question_id, between_all=True) + 1,
                **cls.get_questions_state(new_quiz_state)
            }
        )


class AnswerPostStateCache(QuestionRetrieveCache):
    _answer_set_response_schema = {
        "status": None,
        "choices": [],
        "text": "",
        "file": ""
    }

    class Constraints(QuestionRetrieveCache.Constraints):
        pass

    @classmethod
    def _get_response_schema(cls):
        return copy.deepcopy(cls._answer_set_response_schema)

    @classmethod
    def _set_answer_data(
            cls,
            user_id: int,
            quiz_id: int,
            user_quiz_state: dict,
            question_id: int,
            answer_data: tuple[int, str | int | list, int | None, timezone]
    ) -> None:
        user_quiz_state["answers"][question_id] = answer_data
        cls._set_user_quiz_state(user_id, quiz_id, user_quiz_state)

    @classmethod
    def set_answer_for_choice_questions(
            cls, answer_data, answer_type, user_quiz_state, question_id
    ):
        response = cls._get_response_schema()
        choice_ids = answer_data.get("choice_ids")
        grade = answer_data.get("grade")
        response["status"] = answer_data.get("status")
        response["choices"] = answer_data.get("serialized_data")

        cache_ready_data = answer_type, choice_ids, grade, timezone.now()

        user_id = user_quiz_state.get("user_id")
        quiz_id = user_quiz_state.get("quiz_id")

        cls._set_answer_data(user_id, quiz_id, user_quiz_state, question_id, cache_ready_data)
        return response

    @classmethod
    def set_answer_for_text_questions(
            cls, answer_data, answer_type, user_quiz_state, question_id
    ):
        response = cls._get_response_schema()
        text = answer_data.get("text")
        response["text"] = text

        cache_ready_data = answer_type, text, None, timezone.now()

        user_id = user_quiz_state.get("user_id")
        quiz_id = user_quiz_state.get("quiz_id")

        cls._set_answer_data(user_id, quiz_id, user_quiz_state, question_id, cache_ready_data)
        return response

    @classmethod
    def set_answer_for_file_questions(
            cls, answer_data, answer_type, user_quiz_state, question_id, request
    ):
        response = cls._get_response_schema()
        file_relative_path = answer_data.get("file")

        response["file"] = request.build_absolute_uri(settings.MEDIA_URL + file_relative_path)

        user_id = user_quiz_state.get("user_id")
        quiz_id = user_quiz_state.get("quiz_id")

        abs_file_name = f"{settings.BASE_DIR}/{settings.MEDIA_ROOT}/{file_relative_path}"

        cache_ready_data = (
            answer_type, abs_file_name, None, timezone.now()
        )

        cls._set_answer_data(user_id, quiz_id, user_quiz_state, question_id, cache_ready_data)

        return response

    @classmethod
    def set_answer_to_related_answer_type(cls, request, answer_data, answer_type, user_quiz_state, question_id):
        if answer_data.get("choice_ids") is not None:
            return cls.set_answer_for_choice_questions(
                answer_data, answer_type, user_quiz_state, question_id
            )
        elif answer_data.get("text") is not None:
            return cls.set_answer_for_text_questions(
                answer_data, answer_type, user_quiz_state, question_id
            )
        return cls.set_answer_for_file_questions(
            answer_data, answer_type, user_quiz_state, question_id, request
        )

    @classmethod
    def set_answer_data(cls, request, user_id, quiz_id, question_id):

        user_quiz_state = StateCacheManager.get_user_quiz_state(user_id, quiz_id)
        question_instance = cls.retrieve_question_instance(request, user_quiz_state, question_id)
        answer_cls = question_instance.get_related_answer_model_class()
        if not user_quiz_state.get("show_answer_after_question"):
            answer_data_ser = answer_cls.submitting_serializer_class(
                data=request.data, context={"request": request, "question_id": question_id}
            )

        else:
            answer_data_ser = answer_cls.submitting_serializer_class(
                data=request.data, context={"request": request, "question_id": question_id, "full_detail": True}
            )

        answer_data_ser.is_valid()
        answer_data = answer_data_ser.data["submitted_data"]
        return cls.set_answer_to_related_answer_type(
            request, answer_data, question_instance.answer_type, user_quiz_state, question_id
        )

