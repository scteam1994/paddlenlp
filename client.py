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

import json

import grequests

from paddlenlp.utils.doc_parser import DocParser
from time import time
# Define the document parser
doc_parser = DocParser()

image_paths = ["/home/topnet/PycharmProjects/pythonProject/paddlenlp/applications/information_extraction/document/images/image16.png"]
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
data = {"image": image_base64_docs*128}
t = time()
# Post the requests
req_list = [grequests.post(url=url, headers=headers, data=json.dumps(data))]
res_list = grequests.map(req_list)
#r = grequests.post(url=url, headers=headers, data=json.dumps(data))
print(time()-t)
#print(res_list[0].content.decode("utf-8"))