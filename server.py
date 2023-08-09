# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import base64
import json
import logging
import os

import cv2
import numpy as np
import requests
from pydantic import BaseModel
from paddlenlp import Taskflow
from fastapi import FastAPI
from typing import List

from paddlenlp.utils.doc_parser import DocParser

# The schema changed to your defined schema
schema = ["开票日期", "名称", "纳税人识别号", "开户行及账号", "金额", "价税合计", "No", "税率", "地址、电话", "税额"]
# The task path changed to your best model path
uie1 = Taskflow("information_extraction", schema=["开户行及账号"], model="uie-x-base", layout_analysis=False,
device_id=0,max_seq_len=1024,position_prob=0.3)
# If you want to define the finetuned uie service
app = FastAPI()

class Text(BaseModel):
    '''
    {"imgName":"图片名","imgBase64":"图片base64","imgText": "图片文字","imgPath":"图片绝对路径"}
    '''
    imageName: str
    image: str
    imageText: str

class ImagesItems(BaseModel):
    '''
    {
        "image":[{"imgName": "图片名", "imgUrl": "远程地址", "imgBase64": "base64内容", "imgText": "图片文字", "imgPath": "图片本地路径",
                 "imgData": "发票验真字段内容"}]
    }
    '''
    image: List
@app.post("/uie")
async def predict(images_items: ImagesItems):
# def predict(images_items: ImagesItems):
    '''
    三大合同分类
    :param images_items:
    :return:
    '''
    images = images_items.image
    if images.__len__() == 0:
        return {"code": "A0400", "data": [], "msg": "image is null"}

    image_data = []
    image_name = []
    imagePath = []
    # print(images)
    try:
        for image in images:
            if image.get("imgBase64", ""):
                image_base64 = image.get("imgBase64").replace(' ', '+')
                image_str = base64.b64decode(image_base64)
                # print(image_str)
                image_ndarray = np.frombuffer(image_str, np.uint8)
                img = cv2.imdecode(image_ndarray, cv2.IMREAD_COLOR)  # BGR
                # print(">>>>>>>>>>>>>>>>>>>>>",image)
                image_data.append(img)
                image_name.append(image.get("imgName"))
                imagePath.append(image.get("imgPath"))
            elif image.get("imgPath", ""):
                img_path = image.get("imgPath")
                if not os.path.exists(img_path):
                    return {"code": "A0400", "data": [], "msg": f"{img_path}image is null"}
                img = cv2.imread(img_path)
                image_data.append(img)
                image_name.append(image.get("imgName"))
                imagePath.append(img_path)
            else:
                return {"code": "A0400", "data": [], "msg": "image is null"}
    except Exception as e:
        logging.info(str(e))
        return {"code": "B0001", "data": [], "msg": str(e)}
    results = uie1(image_data)
    return {"code": "00000", "data":json.dump(results), "msg": "success"}

if __name__ == '__main__':
    doc_parser = DocParser()

    image_paths = [
        "/home/topnet/PycharmProjects/pythonProject/paddlenlp/applications/information_extraction/document/images/image16.png"]
    image_base64_docs = []

    # Get the image base64 to post
    for image_path in image_paths:
        req_dict = {}
        doc = doc_parser.parse({"doc": image_path}, do_ocr=False)
        base64 = doc["image"]
        req_dict["doc"] = base64
        image_base64_docs.append(req_dict)

    url = "http://0.0.0.0:8189/uie"
    headers = {"Content-Type": "application/json"}
    data = {"image": image_base64_docs * 128}
    r = predict(data)
    print(r)