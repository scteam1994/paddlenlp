import os
import pickle

import cv2
import numpy as np
from paddleocr import PaddleOCR


class Contract_front_classifier(object):
    def __init__(self,flatten=True):
        self.keywords = [
        ['发包', '承包', '甲方', '乙方', '买方', '卖方', '全称', '承租', '出租', '分包', '建设单位', '施工单位', '转让方',
         '受让方', '转让人', '受让人','出卖','买受'],
        ['(以下简称', '（以下简称'],
        ['第一部分', '第一条', '第一节', '第一章'],
        ['一、', '一。', '一．', '一.'],
        ['协议书', '项目合同书', '采购合同', '销售合同', '买卖合同', '工程合同', '施工合同'],
        ['概况', '概述'],
        ['(盖章', '(签字', '(公章', '(盖单位章', '（盖章', '（签字', '（公章', '（盖单位章', '盖章)', '签字)', '公章)',
         '盖单位章)', '盖章）', '签字）', '公章）''盖单位章）', '(签名', '（签名', '签名）', '签名)', '(印章', '（印章', '印章）',
         '印章)', ],
        ['附件', '附录', '目录', '说明'],
        ['中华人民共和国'],
        ['有限公司'],
        ['代理', '委托'],
        ['法定代表'],
        ['补充'],
        ['自愿', '公平'],
        ['有限公司'],
    ]
        self.keywords_flatten = [self.keywords[i][j] for i in range(len(self.keywords)) for j in range(len(self.keywords[i]))]
        if flatten:
            self.keywords = self.keywords_flatten
        self.model = pickle.load(open('model.pkl','rb'))
    def extract_text(self,ocr_result_dict):

        all_text = []
        if isinstance(ocr_result_dict,dict):
            for img_id,ocr_result in ocr_result_dict.items():
                text = []
                for line in ocr_result[0]:
                    # text.append(re.sub(r, '', line[1][0]))
                    text.append(line[1][0])
                all_text.append(text)
        elif isinstance(ocr_result_dict,list):
            for ocr_result in ocr_result_dict:
                text = []
                for line in ocr_result[0]:
                    # text.append(re.sub(r, '', line[1][0]))
                    text.append(line[1][0])
                all_text.append(text)
        return all_text

    def convert_data_to_logits(self,words_all):
        words_all_flatten = [self.flatten_txt(words_all[i]) for i in range(len(words_all))]
        # words_frq = frequency_count(words_all_flatten)
        data = self.generate_data(words_all_flatten)
        data = np.array(data)
        return data
    def generate_data(self,words_all):
        contains_data = []
        for head_words in words_all:
            contains = np.zeros(len(self.keywords))
            if isinstance(head_words,list):
                for word in head_words:
                    for i,keyword in enumerate(self.keywords):
                        if isinstance(keyword,str):
                            if keyword in word:
                                contains[i] = 1
                        elif isinstance(keyword,list):
                            for kw in keyword:
                                if kw in word:
                                    contains[i] = 1
            elif isinstance(head_words,str):
                for i,keyword in enumerate(self.keywords):
                    if isinstance(keyword,str):
                        if keyword in head_words:
                            contains[i] = 1
                    elif isinstance(keyword,list):
                        for kw in keyword:
                            if kw in head_words:
                                contains[i] = 1
            contains_data.append(contains)
        return contains_data
    def flatten_txt(self,txt_list:list):
        txt = ''
        for line in txt_list:
            if isinstance(line, list):
                txt += self.flatten_txt(line)
            elif isinstance(line, str):
                txt += line
        return txt
    
    def inference(self,data):
        data = self.extract_text(data)
        data = self.convert_data_to_logits(data)
        return self.model.predict(data)


model = Contract_front_classifier()
ocr = PaddleOCR(use_angle_cls=True, lang="ch",use_gpu=True)
ocr_result_dict = {}
test_path = '/home/topnet/图片/house_sale_contract/house_sale_contract'
for k in os.listdir(test_path)[:5]:
    img = cv2.imread(os.path.join(test_path,k))
    ocr_result_dict[k] = ocr.ocr(img)

"""
ocr_result_dict = {
"img1.jpg":ocr.ocr(img1),
"img2.jpg":ocr.ocr(img2),
}
或
ocr_result_dict = [
ocr.ocr(img1),ocr.ocr(img2),
]

"""
def test(ocr_result_dict):
    res = model.inference(ocr_result_dict)
    print(res)