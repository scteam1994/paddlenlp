import os
import re

import numpy as np
from PIL import Image

from paddlenlp import Taskflow
from paddleocr import PaddleOCR, draw_ocr

# result = [
#     {
#         'prompt': '图片上方购买方的称',
#         'result': [{'value': '绿培集团安庆置业有限公司',
#                     'prob': 1.0,
#                     'start': 27,
#                     'end': 38},
#                    {'value': '日',
#                     'prob': 0.0,
#                     'start': 26,
#                     'end': 26}
#                    ],
#         'bbox': [
#             [90.0, 108.0, 181.2, 118.0],
#             [439.0, 86.0, 447.0, 103.0]
#         ]
#     },
#     {
#         'prompt': '图片下方销售方的名称',
#         'result': [{'value': '上购新光工程咨询有限会司',
#                     'prob': 0.94,
#                     'start': 187,
#                     'end': 198},
#                    {'value': '绿培集团安庆置业有限公司',
#                     'prob': 0.28,
#                     'start': 27,
#                     'end': 38}
#                    ],
#         'bbox': [
#             [95.0, 277.0, 183.8, 287.0],
#             [90.0, 108.0, 181.2, 118.0]
#         ]
#     },
#
# ]

def draw_det_res(ocr_result, img):
    font_path = "./simfang.ttf"
    image = Image.open(img.get("doc")).convert('RGB')
    boxes = [line[0] for line in ocr_result[0]]
    # txts = [line[1][0] for line in ocr_result[0]]
    # scores = [line[1][1] for line in ocr_result[0]]
    im_show = draw_ocr(image, boxes, txts=None, scores=None, font_path=font_path)
    im_show = Image.fromarray(im_show)
    base = os.path.basename(img.get("doc"))
    temp = os.path.splitext(base)
    img_url = img.get("doc").replace(base, temp[0] + "_det_ori" + temp[1])
    im_show.save(img_url)


class Postprocessor(object):
    def __init__(self):
        with open("db.txt", "r", encoding="utf-8") as f:
            self.db = f.readlines()
        self.db = [i.split('\t') for i in self.db]
        self.prompt_combine = {}
        for d in self.db:
            for p in d[2].split(' '):
                self.prompt_combine[p] = [d[0], d[1], d[3]]
        self.code = {}
        self.annotation = {}
        self.is_important = {}
        for prompt in self.prompt_combine:
            self.code[prompt] = self.prompt_combine[prompt][1]
            self.annotation[prompt] = self.prompt_combine[prompt][0]
            self.is_important[prompt] = self.prompt_combine[prompt][2]



    def __call__(self, result, thresh_hold=0.2):
        for r in result:
            if len(r['bbox']) < 1:
                continue
            is_over_thresh = [r['result'][i]['prob'] > thresh_hold for i in range(len(r['result']))]
            for i in range(len(is_over_thresh)):
                if not is_over_thresh[i]:
                    r['result'].pop(i)
                    r['bbox'].pop(i)
            code = self.code[r['prompt']]
            if len(r['result']) > 1:
                if hasattr(self,code):
                    code = self.code[r['prompt']]
                    r = self.__getattribute__(code)(r)
        return result

    def purchaserName(self, r):
        # find purchaserName in invoice
        box1 = np.array(r['bbox'][0])
        box2 = np.array(r['bbox'][0])
        upper = np.argmin(((np.mean((box1[1], box1[3]))), np.mean((box2[1], box2[3]))))
        r['result'] = [r['result'][upper]]
        r['bbox'] = [r['bbox'][upper]]
        return r

    def sellerName(self, r):
        # find sellerName in invoice
        box1 = np.array(r['bbox'][0])
        box2 = np.array(r['bbox'][1])
        lower = np.argmax(((np.mean((box1[1], box1[3]))), np.mean((box2[1], box2[3]))))
        r['result'] = [r['result'][lower]]
        r['bbox'] = [r['bbox'][lower]]
        return r

    def purchaserBank(self, r):
        # find purchaserBank in invoice
        box1 = np.array(r['bbox'][0])
        box2 = np.array(r['bbox'][1])
        upper = np.argmin(((np.mean((box1[1], box1[3]))), np.mean((box2[1], box2[3]))))
        r['result'] = [r['result'][upper]]
        r['bbox'] = [r['bbox'][upper]]
        return r

    def sellerBank(self, r):
        # find sellerBank in invoice
        box1 = np.array(r['bbox'][0])
        box2 = np.array(r['bbox'][1])
        lower = np.argmax(((np.mean((box1[1], box1[3]))), np.mean((box2[1], box2[3]))))
        r['result'] = [r['result'][lower]]
        r['bbox'] = [r['bbox'][lower]]
        return r

docprompt = Taskflow("document_intelligence", topn=2)
result = docprompt([{"doc": "./image14.png",
                "prompt": ["图片上方购买方的称", "图片下方销售方的名称"]}])
p = Postprocessor()
result = p(result)

print(result)
