import logging

from flask import Flask, request, jsonify

from chatbot.bot import Bot
import cv2
import numpy as np

from image.pill_feature_extractor import PillFeatureExtractor

app = Flask(__name__)
app.logger.setLevel(logging.DEBUG)
log = logging.getLogger(__name__)
FORMAT = "[%(filename)s:%(lineno)s - %(funcName)28s() ] %(message)s"
logging.basicConfig(format=FORMAT)


@app.route('/')
def index():
    return 'Doc-di AI server is now on FIRE! 🔥🔥🔥'


@app.route('/chat', methods=['POST'])
def chat():
    req: dict = request.get_json(silent=True)
    log.debug(req)

    if not req:
        return jsonify({'error': 'No input data'}), 400

    if 'sender' not in req:
        return jsonify({'error': 'Missing parameter: sender'}), 400
    if 'message' not in req:
        return jsonify({'error': 'Missing parameter: message'}), 400

    message = req.get('message', '').strip()
    if not message:
        return jsonify({f'error': f'Invalid parameter: message is falsy(empty or e.t.c.)'}), 400

    sender = req['sender']

    bot = Bot.get_instance()
    response = bot.get_chat_response(sender, message)

    return response


@app.route('/sum', methods=['POST'])
def summarize():
    req: dict = request.get_json(silent=True)
    log.debug(req)

    if not req:
        return jsonify({'error': 'No input data'}), 400

    if 'sender' not in req:
        return jsonify({'error': 'Missing parameter: sender'}), 400
    if 'data' not in req:
        return jsonify({'error': 'Missing parameter: data'}), 400
    if 'query' not in req:
        return jsonify({'error': 'Missing parameter: query'}), 400

    sender = req['sender']
    query = req['query']
    data = req.get('data') or []

    log.debug(data)

    if not data:
        return jsonify({f'error': f'Invalid parameter: data is falsy(empty or e.t.c.)'}), 400

    if not query:
        return jsonify({f'error': f'Invalid parameter: query is falsy(empty or e.t.c.)'}), 400

    for i, datum in enumerate(data):
        if 'title' not in datum:
            return jsonify({f'error': f'Missing parameter: data[{i}].title'}), 400
        if 'link' not in datum:
            return jsonify({f'error': f'Missing parameter: link[{i}].title'}), 400
        if not datum['title'].strip():
            return jsonify({f'error': f'Missing parameter: data[{i}].title'}), 400
        if not datum['link'].strip():
            return jsonify({f'error': f'Missing parameter: link[{i}].title'}), 400

    bot = Bot.get_instance()
    response = bot.get_summary_response(sender, data)

    return response


@app.route('/pill', methods=['POST'])
def extract_pill_features():
    log.debug(f'Content-Type: {request.content_type}')

    try:
        extractor = PillFeatureExtractor.get_instance()
        if request.content_type.startswith('image/'):
            image = np.fromstring(request.data, dtype=np.uint8)
            image = cv2.imdecode(image, cv2.IMREAD_COLOR)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            result = extractor.extract(image)
        elif request.content_type == 'application/json':
            req: dict = request.get_json(silent=True)
            log.debug(req)
            result = extractor.extract_with_path(req['image_path'])
        else:
            return {f'error': f'Invalid parameter: data not included'}, 400
    except Exception as e:
        log.error(e)
        return {f'error': f'Unhandled exception occured: {e}'}, 500
    else:
        return result


if __name__ == '__main__':
    print('test')