from flask import Flask, request, render_template, jsonify
import requests
import numpy as np
import base64
from PIL import Image
import io
import asyncio
import aiohttp
import http.client
import time
from concurrent.futures import ThreadPoolExecutor
import json

from flask_cors import CORS

app = Flask(__name__)
CORS(app)

app.secret_key = 'GimmeShelter'

http.client._MAXLINE = 655360

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'PNG'}


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


async def get_predictions(session, url, data, headers):
    async with session.post(url, data=data, headers=headers) as resp:
        if resp.status != 200:
            return 500
        else:
            preds = await resp.read()
            preds = np.asarray(json.loads(preds))
            preds = np.argmax(preds)
            return preds


@app.route('/', methods=["GET"])
def index():
    return render_template("index.html")


@app.route('/predict', methods=["POST"])
async def value_post():
    # 1- file checking
    tic = time.time()
    start_time = time.time()
    files = {}
    loaded_files = request.files.getlist('image')
    pool = ThreadPoolExecutor(max_workers=2)
    for idx, r in enumerate(loaded_files):
        if not (allowed_file(r.filename)):
            message = {'text': 'Extension not allowed. Insert only png or jpg/jpeg images!'}
            return render_template("index_error.html", api_response=message)
        else:
            files[f"image{idx}"] = r
    pool.shutdown(wait=True)
    print(f"allowed file checking: {time.time() - start_time}")
    # 2- async Inference
    ret = {"results": []}
    async with aiohttp.ClientSession(trust_env=True) as session:
        tasks = []
        for idx, (k, file) in enumerate(files.items()):
            # write return obj
            ret_obj = {'filename': file.filename, 'prediction': 0}
            original = Image.open(file).convert("L")
            original = original.resize((28, 28), Image.BILINEAR)
            ret["results"].append(ret_obj)
            # prepare post call
            buffer = io.BytesIO()
            original.save(buffer, format='PNG')
            buffer.seek(0)
            payload = buffer.read()
            url = "http://worker:8080/predictions/mymodel"
            headers = {'content-type': 'image/gif'}
            tasks.append(
                asyncio.ensure_future(get_predictions(session=session, url=url, data=payload, headers=headers)))
        start_time = time.time()
        predictions = await asyncio.gather(*tasks)
        print(f"only inference time: {time.time() - start_time}")
        for idx, pred in enumerate(predictions):
            ret["results"][idx]['prediction'] = int(pred)
    toc = time.time()
    print(f"Inference Total Timer: {toc - tic}")
    return jsonify(ret)


if __name__ == '__main__':
    app.run(debug=True, port=5002)