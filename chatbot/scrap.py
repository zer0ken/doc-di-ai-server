import re

import requests
from bs4 import BeautifulSoup

headers = {
    'User-Agent': 'Mozilla/5.0 '
                  '(Windows NT 10.0; Win64; x64)'
                  'AppleWebKit/537.36 (KHTML, like Gecko) '
                  'Chrome/73.0.3683.86 Safari/537.36'
}


def extract(soup, selector):
    selected = soup.select(selector)
    texts = []
    for tag in selected:
        text = tag.get_text().strip()
        if text:
            texts.extend(text.split('###'))
    answer = ' '.join(texts)
    answer = re.sub(r'\s+', ' ', answer)
    return answer


def scrap_q_and_a(url: str) -> tuple[str, str]:
    data = requests.get(url, headers=headers)
    soup = BeautifulSoup(data.text, 'html.parser')

    delimiter = '###'
    for line_break in soup.findAll('br'):
        line_break.replaceWith(delimiter)

    question_selector = '#content div.questionDetail'
    answer_selector = '#answer_1 div.answerDetail'

    question = extract(soup, question_selector)
    answer = extract(soup, answer_selector)

    return question.strip(), answer.strip()


if __name__ == '__main__':
    url = r'https://kin.naver.com/qna/detail.naver?d1id=7&dirId=7010104&docId=473478380&qb=MzDrjIAg7Jes7J6QIOyngOyGjeyggeyduCDrkZDthrUg7LmY66OM&enc=utf8&section=kin.qna&rank=48&search_sort=0&spq=0'
    # url = r'https://kin.naver.com/qna/detail.naver?d1id=7&dirId=70301&docId=477002203&qb=65GQ7Ya1&enc=utf8&section=kin.qna&rank=1&search_sort=0&spq=0'
    q, a = scrap_q_and_a(url)
    print(q)
    print(a)
