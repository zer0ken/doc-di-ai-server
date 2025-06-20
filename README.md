<img src="https://img.shields.io/badge/python-3776AB?logo=python&logoColor=white"> <img src="https://img.shields.io/badge/Flask-000000?logo=flask&logoColor=white"> <img src="https://img.shields.io/badge/Yolov11-111F68?logo=yolo&logoColor=white"> <img src="https://img.shields.io/badge/ResNet--18-EE4C2C?logo=pytorch&logoColor=white"> <img src="https://img.shields.io/badge/Google%20Gemini-8E75B2?logo=googlegemini&logoColor=white">

# `doc-di-ai-server`
이 프로젝트는는 똑디 앱의 백엔드 서버의 일부로서, 마이크로서비스 구조에서 동작하는 AI 기반 서비스입니다. Flask 프레임워크를 사용하여 챗봇, 요약, 알약 이미지 분석 등의 기능을 제공합니다.

## 주요 기능

- **챗봇 서비스**  
  사용자 메시지에 대해 챗봇이 답변을 제공합니다.

- **텍스트 요약**  
  주어진 데이터와 쿼리에 대해 요약 결과를 반환합니다.

- **알약 이미지 특징 추출**  
  업로드된 알약 이미지를 분석하여 특징을 추출합니다.

## API 엔드포인트

### 1. 서버 상태 확인

- **GET /**  
  서버가 정상 동작 중인지 확인할 수 있습니다.  
  **Response:**  
  ```
  Doc-di AI server is now on FIRE! 🔥🔥🔥
  ```

### 2. 챗봇

- **POST /chat**  
  사용자의 메시지에 대해 챗봇이 답변합니다.
  - **Request (JSON):**
    ```json
    {
      "sender": "user_id",
      "message": "질문 내용"
    }
    ```
  - **Response:**  
    챗봇의 답변(JSON)

### 3. 요약

- **POST /sum**  
  여러 데이터와 쿼리를 입력하면 요약 결과를 반환합니다.
  - **Request (JSON):**
    ```json
    {
      "sender": "user_id",
      "query": "요약할 쿼리",
      "data": [
        {"title": "제목", "link": "링크"},
        ...
      ]
    }
    ```
  - **Response:**  
    요약 결과(JSON)

### 4. 알약 특징 추출

- **POST /pill**
  - 이미지 파일(바이너리) 또는 JSON(`{"image_path": "경로"}`)로 알약 이미지를 전달하면 특징을 추출하여 반환합니다.

## 실행 방법

1. 의존성 설치
    ```bash
    pip install -r requirements.txt
    ```

2. 서버 실행
    ```bash
    python app.py
    ```

3. (테스트용 대화)  
   `__main__` 블록을 활용해 직접 챗봇 응답을 확인할 수 있습니다.

## 프로젝트 구조

- `app.py`: Flask 서버 및 주요 엔드포인트 정의
- `chatbot/bot.py`: 챗봇 로직
- `image/pill_feature_extractor.py`: 알약 이미지 특징 추출

## 환경

- Python
- Flask
