import json
import logging
import os
from time import sleep, time

from dotenv import load_dotenv
import google.generativeai as genai
from google.api_core.exceptions import ResourceExhausted
from google.generativeai import GenerativeModel, ChatSession
from google.generativeai.types import HarmCategory, HarmBlockThreshold, BlockedPromptException, StopCandidateException

from chatbot.scrap import scrap_q_and_a
from chatbot.system_instruction import CHATBOT_SYSTEM_INSTRUCTION, SUMMARIZER_SYSTEM_INSTRUCTION
from chatbot.tools import tools
from chatbot.cleanup import cleanup_chat_response, cleanup_summary_response

load_dotenv()
genai.configure(api_key=os.getenv('GENAI_API_KEY'))

log = logging.getLogger(__name__)


class Bot:
    _INSTANCE = None
    FILTERED_CONTENT = '유해한 컨텐츠가 감지되어 메시지가 검열되었습니다. 불편을 끼쳐드려 죄송합니다. 다른 도움이 필요한 부분이 있을까요?'

    @classmethod
    def get_instance(cls):
        if cls._INSTANCE is None:
            cls._INSTANCE = Bot()
        return cls._INSTANCE

    @classmethod
    def get_filtered_response(cls, sender_id: str) -> list[dict]:
        return [{'recipient_id': sender_id, 'text': Bot.FILTERED_CONTENT}]

    def __init__(self):
        self.chat_model = GenerativeModel(
            model_name="gemini-1.5-flash-001",
            tools=tools,
            system_instruction=CHATBOT_SYSTEM_INSTRUCTION,
            safety_settings={
                HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
            }
        )
        self.sum_model = GenerativeModel(
            model_name="gemini-1.5-flash",
            system_instruction=SUMMARIZER_SYSTEM_INSTRUCTION
        )
        self.sessions: dict[str, ChatSession] = {}
        self.timestamp: dict[str, float] = {}

    def get_session(self, sender_id: str) -> ChatSession:
        if sender_id not in self.sessions:
            self.sessions[sender_id] = self.chat_model.start_chat(enable_automatic_function_calling=True)
            self.timestamp[sender_id] = time()
        return self.sessions[sender_id]

    def get_chat_response(self, sender_id: str, prompt: str) -> list[dict]:
        log.debug(f'{sender_id}: {prompt}')

        chat = self.get_session(sender_id)
        prompt = prompt.strip()

        text_response = []

        gen_time = time()
        while True:
            try:
                responses = chat.send_message(prompt)
                for chunk in responses:
                    text_response.append(chunk.text)
                break
            except ResourceExhausted as e:
                log.debug(e)
                log.debug('... sleep for 0.5 sec.')
                sleep(0.5)
            except (BlockedPromptException, StopCandidateException) as e:
                return Bot.get_filtered_response(sender_id)
        gen_time = time() - gen_time

        result = "".join(text_response).strip()
        self.timestamp[sender_id] = time()

        log.debug(f'{sender_id}: {result}')
        result = cleanup_chat_response(result, sender_id)

        log.debug(f'{result}\n... took {gen_time} sec')
        return result

    def get_summary_response(self, sender_id: str, data: list[dict]) -> list[dict]:
        web_data = []
        for datum in data:
            q, a = scrap_q_and_a(datum['link'])
            if q and a:
                web_data.append({'title': datum['title'], 'link': datum['link'], 'question': q, 'answer': a})
        text_response = []

        gen_time = time()
        while True:
            try:
                responses = self.sum_model.generate_content(json.dumps(web_data, ensure_ascii=False), stream=True)
                for chunk in responses:
                    text_response.append(chunk.text)
                break
            except ResourceExhausted as e:
                log.debug(e)
                log.debug('... sleep for 0.5 sec.')
                sleep(0.5)
            except (BlockedPromptException, StopCandidateException) as e:
                return Bot.get_filtered_response(sender_id)
        gen_time = time() - gen_time

        result = "".join(text_response).strip()
        result = cleanup_summary_response(result, sender_id)

        log.debug(f'{result}\n... took {gen_time} sec')
        return result


if __name__ == '__main__':
    test_data = [
        {
            'title': r'두통 심할 때 어떻게 견디세요?(논현 두통)',
            'link': f'https://kin.naver.com/qna/detail.naver?d1id=7&dirId=70301&docId=477002203&qb=65GQ7Ya1&enc=utf8&section=kin.qna&rank=1&search_sort=0&spq=0'
        }
        for _ in range(10)
    ]
    Bot.get_instance().get_summary_response('test001', test_data)

# todo: expire idle session automatically
