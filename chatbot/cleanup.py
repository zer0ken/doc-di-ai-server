import json
import logging
from typing import Any, Dict

from chatbot.instructions.problem_literals import INVALID_SHAPE, INVALID_COLOR, EMPTY_AGE, EMPTY_GENDER, EMPTY_IMPRINT
from chatbot.tools import validate_pill_shape, validate_pill_color, validate_age, validate_gender

log = logging.getLogger()


def remove_markdown(text):
    while '```json' in text:
        text = text.replace('```json', '')
    while '```' in text:
        text = text.replace('```', '')
    return text.strip()


def cleanup_chat_response(response: str, recipient_id: str) -> tuple[dict, list[str]]:
    response = remove_markdown(response)

    """json 문자열을 파이썬 자료구조로 파싱"""
    response_object = json.loads(response)

    """주요 값들을 추출"""
    custom = response_object.get('custom', None) or {}
    action = custom.get('action', None)
    data = custom.get('data', None) or {}

    shape = data.get('shape', None) or ''
    color = data.get('color', None) or ''
    imprint = data.get('imprint', None) or ''

    query = data.get('query', None) or ''
    age = data.get('age', None) or ''
    gender = data.get('gender', None) or ''

    done = (data.get('done', None) or '') == '완성됨'

    """유효성 검사"""
    has_form = True
    problems = []

    if not custom or action is None or not data:
        has_form = False
    elif action == 'DB_SEARCH':
        if not shape or not color:
            has_form = False
        elif not imprint:
            has_form = False
            problems.append(EMPTY_IMPRINT)
        if not validate_pill_shape(shape):
            has_form = False
            problems.append(INVALID_SHAPE)
        elif not validate_pill_color(color):
            has_form = False
            problems.append(INVALID_COLOR)
    elif action == 'WEB_SEARCH':
        if not validate_age(age):
            has_form = False
            problems.append(EMPTY_AGE)
        elif not validate_gender(gender):
            has_form = False
            problems.append(EMPTY_GENDER)

    if not has_form and custom:
        del response_object['custom']

    """불필요한 값 제거"""
    if has_form:
        if action == 'DB_SEARCH':
            for key in ('query', 'age', 'gender'):
                if key in data:
                    del data[key]
            if imprint == '없음':
                del data['imprint']
        elif action == 'WEB_SEARCH':
            for key in ('shape', 'color', 'imprint'):
                if key in data:
                    del data[key]
            if gender in query:
                data['query'] = data['query'].replace(gender, '')
            if age in query:
                data['query'] = data['query'].replace(age, '')
            if gender not in ('거부함', '생략됨') :
                data['query'] = data['query'] + ' ' + gender
            if age not in ('거부함', '생략됨'):
                data['query'] = data['query'] + ' ' + age
            del data['age']
            del data['gender']
            data['query'] = data['query'].strip()

    response_object['recipient_id'] = recipient_id
    if 'done' in data:
        del data['done']

    return response_object, problems if done else []


def cleanup_summary_response(response: str, recipient_id: str) -> dict:
    print(f'@CLEANUP_SUM: {response}')
    response = remove_markdown(response)
    print(f'>>> {response}')

    response_object_ = json.loads(response)
    # valid_summaries = []

    summary = response_object_['summary']
    # data = response_object_['data']
    #
    # for datum in data:
    #     title = datum.get('title', None) or ''
    #     link = datum.get('title', None) or ''
    #     question = datum.get('question', None) or ''
    #     answer = datum.get('answer', None) or ''
    #
    #     if not title or not link or not question or not answer:
    #         continue
    #     # if len(question) > 150 or len(answer) >= 500:
    #     #     continue
    #
    #     valid_summaries.append(datum)

    response_object = {
        'recipient_id': recipient_id,
        'text': summary,
        # 'data': valid_summaries,
    }

    return response_object
