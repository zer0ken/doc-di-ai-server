import json
import logging
import os
from time import sleep, time

from dotenv import load_dotenv
import google.generativeai as genai
from google.api_core.exceptions import ResourceExhausted
from google.generativeai import GenerativeModel, ChatSession
from google.generativeai.types import HarmCategory, HarmBlockThreshold, BlockedPromptException, StopCandidateException
from jinja2.compiler import generate

from chatbot.scrap import scrap_q_and_a
from chatbot.tools import tools
from chatbot.cleanup import cleanup_chat_response, cleanup_summary_response

load_dotenv()
genai.configure(api_key=os.getenv('GENAI_API_KEY'))

log = logging.getLogger()


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
        with open('chatbot/instructions/chatbot.md', encoding='utf-8', mode='r') as f:
            chatbot_instruction = '\n'.join(f.readlines())

        self.chat_model = GenerativeModel(
            model_name="gemini-1.5-flash",
            tools=tools,
            system_instruction=chatbot_instruction,
            safety_settings={
                HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
            }
        )

        with open('chatbot/instructions/summary.md', encoding='utf-8', mode='r') as f:
            summary_instruction = '\n'.join(f.readlines())

        self.sum_model = GenerativeModel(
            model_name="gemini-1.5-flash",
            system_instruction=summary_instruction
        )
        self.sessions: dict[str, ChatSession] = {}
        self.timestamp: dict[str, float] = {}

    def get_session(self, sender_id: str) -> ChatSession:
        if sender_id not in self.sessions:
            self.sessions[sender_id] = self.chat_model.start_chat(enable_automatic_function_calling=True)
            self.timestamp[sender_id] = time()
        return self.sessions[sender_id]

    def get_chat_response(self, sender_id: str, prompt: str) -> list[dict]:
        def generate_response(prompt_):
            text_response = []
            while True:
                try:
                    responses = chat.send_message(prompt_)
                    for chunk in responses:
                        text_response.append(chunk.text)
                    break
                except ResourceExhausted as e:
                    log.debug(e)
                    log.debug('... sleep for 0.5 sec.')
                    sleep(0.5)
                except (BlockedPromptException, StopCandidateException) as e:
                    return Bot.FILTERED_CONTENT
            return '\n'.join(text_response).strip()


        print(f'{sender_id}: {prompt}')

        chat = self.get_session(sender_id)
        prompt = prompt.strip()


        gen_time = time()
        result = generate_response(prompt)
        gen_time = time() - gen_time

        self.timestamp[sender_id] = time()

        print(f'# 생성된 응답\n{result}')
        result, problems = cleanup_chat_response(result, sender_id)

        problems_prompt = ',\n\t'.join(problems)

        print(f'{result}\n... took {gen_time} sec')

        if problems:
            print(f'... 발견된 문제: [\n\t{problems_prompt}\n]')
            gen_time = time()
            result = generate_response(problems_prompt)
            gen_time = time() - gen_time
            print(f'# 문제 해결을 위해 재생성된 응답\n{result}')
            result, problems = cleanup_chat_response(result, sender_id)
            print(f'{result}\n... took {gen_time} sec')

        return result

    def get_summary_response(self, request) -> list[dict]:
        sender_id = request['sender']
        data = request['data']

        web_data = []
        for datum in data:
            q, a = scrap_q_and_a(datum['link'])
            if q and a:
                web_data.append({'title': datum['title'], 'link': datum['link'], 'question': q, 'answer': a})

        request['data'] = web_data

        gen_time = time()
        while True:
            try:
                response = self.sum_model.generate_content(json.dumps(request, ensure_ascii=False))
                break
            except ResourceExhausted as e:
                log.debug(e)
                log.debug('... sleep for 0.5 sec.')
                sleep(0.5)
            except (BlockedPromptException, StopCandidateException) as e:
                return Bot.get_filtered_response(sender_id)
        gen_time = time() - gen_time

        result = response.text.strip()
        result = cleanup_summary_response(result, sender_id)

        log.debug(f'{result}\n... took {gen_time} sec')
        return result

