import os
import pickle
import shutil

import cv2
import numpy as np
from paddleocr import PaddleOCR

from statistic import keywords
class Contract_front_classifier(object):
    def __init__(self,flatten=True):
    #     self.keywords = [
    #     [['all'],'销售合同','供货合同','购销合同','承揽合同','销购合同','采购协议','供应合同','定做合同','加工合同'],
    #     [['all'],'甲方', '乙方', '买方', '卖方'],
    #     [['all'],'发包', '承包', '全称', '承租', '出租', '分包', '建设单位', '施工单位','转让方','受让方','转让人','受让人','出卖','买受','供方','需方','委托人','代建人','委托方','代建方','代理人','代理方',],
    #     [['all'],'地点','时间','签订'],
    #     [['all'],'(以下简称', '（以下简称'],
    #     [['all'],'第一部分', '第一条', '第一节', '第一章'],
    #     [['all'],'一、', '一。', '一．', '一.'],
    #     [['all'], '协议'],
    #     [['all'], '项目合同书', '采购合同', '销售合同', '买卖合同', '工程合同', '施工合同','合同条款','分包合同','安装合同','服务合同','维护合同','承包合同'],
    #     [['all'],'概况', '概述'],
    #     [['all'],'(盖章', '(签字', '(公章', '(盖单位章','（盖章', '（签字', '（公章','（盖单位章', '盖章)', '签字)', '公章)','盖单位章)', '盖章）', '签字）', '公章）''盖单位章）','(签名','（签名','签名）','签名)','(印章','（印章','印章）','印章)',],
    #     [['all'],'附件','附录','目录','说明'],
    #     [['all'],'中华人民共和国'],
    #     [['all'],'法定代表'],
    #     [['all'],'补充'],
    #     [['all'],'自愿','公平'],
    # ]
        self.keywords = keywords
        self.keywords_flatten = []
        for i in self.keywords:
            word_limit = i[0]
            for j in word_limit:
                for k in i[1:]:
                    if isinstance(k, str):
                        self.keywords_flatten.append([j, k])
        if flatten:
            self.keywords = self.keywords_flatten
        self.model = pickle.load(open('contract_front_classify/model_loc.pkl','rb'))
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
            loc = []
            contains = np.zeros(len(self.keywords))
            if isinstance(head_words,list):
                for word in head_words:
                    for i,keyword in enumerate(self.keywords):
                        if isinstance(keyword,str):
                            if keyword in word:
                                contains[i] = word.count(keyword)
                        elif isinstance(keyword, list):
                            for kw in keyword:
                                if kw in word:
                                    contains[i] = word.count(keyword)
            elif isinstance(head_words, str):
                for i, keyword in enumerate(self.keywords):
                    if isinstance(keyword, str):
                        if keyword in head_words:
                            contains[i] = head_words.count(keyword)
                    elif isinstance(keyword, list):
                        # for kw in keyword:
                        #     if kw in verf_data:
                        #         contains[i] = 1
                        if "all" == keyword[0]:
                            verf_data = head_words
                            for kw in keyword[1:]:
                                if kw in verf_data:
                                    contains[i] = 1
                                    loc.append(head_words.find(kw) / len(head_words))
                                else:
                                    loc.append(-1)
                        elif type(keyword[0]) == int:
                            verf_data = head_words[:keyword[0]]
                            for kw in keyword[1:]:
                                if kw in verf_data:
                                    contains[i] = verf_data.count(kw)
                        else:
                            for kw in keyword:
                                if kw in head_words:
                                    contains[i] = head_words.count(kw)
            contains_data.append(np.hstack((contains, np.array(loc))))
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
ocr = PaddleOCR(use_angle_cls=True, lang="ch",use_gpu=True,ocr_version="PP-OCRv4")

test_path = '/home/topnet/图片/house_sale_contract/3'
test_path = '/home/topnet/图片/media'

imglist = [
    '2023072514_a61fd153d19e4e90ccb05eff641d4a6f.png' ,
    'image108.jpeg'                                   ,
    '6b494a8e8e64a949dca8499c92145f56.jpeg'           ,
    '2023072514_f4be853beaa4552f2663043847646f71.png' ,
    'image113.jpeg'                                   ,
    '2023072514_1088abfc19133e3692a78a68d4e3b171.png' ,
    'image139.jpeg'                                   ,
    'image154.jpeg'                                   ,
    'image24.png'                                     ,
    '2023060616_d690ad24ae395bdf0e0ec50cd4fed442.jpeg',
    '2023072514_ea5d08a4b98560a728174ee5c058ecb3.png' ,
           ]
for k in sorted(os.listdir(test_path)):
# for k in imglist:
    ocr_result_dict = {}
    img = cv2.imread(os.path.join(test_path,k))
    if not isinstance(img,np.ndarray):
        continue
    ocr_result_dict[k] = ocr.ocr(img)
    res = model.inference(ocr_result_dict)
    print(res)
    # if res[0] == 1:
    #     shutil.move(os.path.join(test_path,k),os.path.join('/home/topnet/图片/设备购置合同/head',k))
    # elif res[0] == 2:
    #     shutil.move(os.path.join(test_path,k),os.path.join('/home/topnet/图片/设备购置合同/tail',k))
    cv2.putText(img,str(res[0]),(100,100),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),5)
    cv2.imshow(k,img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
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
# def test(ocr_result_dict):
#     res = model.inference(ocr_result_dict)
#     print(res)
