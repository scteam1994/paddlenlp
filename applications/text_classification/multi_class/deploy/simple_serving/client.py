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
import shutil

import requests

url = "http://0.0.0.0:8027/taskflow/cls"
headers = {"Content-Type": "application/json"}

if __name__ == "__main__":
    with open("/home/topnet/PycharmProjects/keywords_class/contract_front_classify/all.txt", "r", encoding="utf-8") as f:
        texts = []
        label = []
        for line in f:
            items = line.strip().split("\t")
            if items[1] in ["head", "other"]:
                texts.append(items[0])
                label.append(items[1])
    data = {"data": {"text": texts}}
    r = requests.post(url=url, headers=headers, data=json.dumps(data))
    result_json = json.loads(r.text)
    for i in range(len(result_json["result"])):
        if result_json["result"][i]['predictions'][0]['label'] != label[i]:
            print('####################')
            print(f'预测结果：{result_json["result"][i]["predictions"][0]["label"]}------->真实结果：{label[i]}')



