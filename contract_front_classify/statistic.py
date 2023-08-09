import json
import os
import re
from sklearn import svm
import numpy as np

import cv2
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from sklearn.metrics import accuracy_score
import pickle
import tensorflow as tf

import jieba
def frequency_count(txt):
    words = jieba.lcut(txt)
    counts = {}
    for word in words:
        if len(word) == 1:
            continue
        else:
            counts[word] = counts.get(word,0) + 1
    counts = {k: v/len(txt) for k, v in counts.items()}
    res=sorted(counts.items(),key=lambda x:x[1],reverse=True)
    return res
def extract_text(root):
    import paddle
    from paddleocr import PaddleOCR
    ocr = PaddleOCR(use_angle_cls=True, lang="ch", ocr_version="PP-OCRv3")
    ocr.text_recognizer.rec_batch_num = 16
    img_paths = os.listdir(root)
    # remove comma

    all_text = {}
    for img_path in tqdm(img_paths):
        text = []
        img = cv2.imread(os.path.join(root, img_path))
        ocr_result = ocr.ocr(img, rec=True, cls=False)
        for line in ocr_result[0]:
            # text.append(re.sub(r, '', line[1][0]))
            text.append(line[1][0])
        all_text[img_path] = text
        paddle.device.cuda.empty_cache()
    return all_text


def read_image_txt(root, folders,img_size,use_img=True):
    datas = []
    img_datas = []
    for folder in folders:
        imgs = []
        if os.path.exists(os.path.join(root, folder + '.txt')):
            data = json.load(open(os.path.join(root, folder + '.txt'), 'r'))
        else:
            data = extract_text(os.path.join(root, folder))
            with open(os.path.join(root, folder + '.txt'), 'w') as f:
                f.write(json.dumps(data, ensure_ascii=False, indent=4))
        datas.append(data)
        if not use_img:
            imgs = [np.zeros((1))]*len(data)
        else:
            for img_path in data.keys():
                img = cv2.imread(os.path.join(root, folder, img_path))
                img = normalize(img,img_size)
                imgs.append(img)
        img_datas.append(imgs)
    return (datas, img_datas)

def normalize(img,img_size):
    img = resize_with_pad(img, img_size)
    img = img / 255.0
    return img

def resize_with_pad(img, target_size):
    h, w = img.shape[:2]
    ratio = target_size / max(h, w)
    resized = cv2.resize(img, (int(w * ratio), int(h * ratio)))
    h, w = resized.shape[:2]
    pad_h = (target_size - h) // 2
    pad_h2 = target_size - h - pad_h
    pad_w = (target_size - w) // 2
    pad_w2 = target_size - w - pad_w
    padded = cv2.copyMakeBorder(resized, pad_h, pad_h2, pad_w, pad_w2, cv2.BORDER_CONSTANT, value=[0, 0, 0])
    return padded
def flatten_txt(txt_list:list):
    txt = ''
    for line in txt_list:
        if isinstance(line, list):
            txt += flatten_txt(line)
        elif isinstance(line, str):
            txt += line
    return txt

def generate_data(keywords,words_all):
    contains_data = []
    for head_words in words_all:
        contains = np.zeros(len(keywords))
        if isinstance(head_words,list):
            for word in head_words:
                for i,keyword in enumerate(keywords):
                    if isinstance(keyword,str):
                        if keyword in word:
                            contains[i] = 1
                    elif isinstance(keyword,list):
                        for kw in keyword:
                            if kw in word:
                                contains[i] = 1
        elif isinstance(head_words,str):
            for i,keyword in enumerate(keywords):
                if isinstance(keyword,str):
                    if keyword in head_words:
                        contains[i] = 1
                elif isinstance(keyword,list):
                    for kw in keyword:
                        if kw in head_words:
                            contains[i] = 1
        contains_data.append(contains)
    return contains_data


def convert_data_to_logits(keywords,data):
    words_all = list(data.values())
    words_all_flatten = [flatten_txt(words_all[i]) for i in range(len(words_all))]
    # words_frq = frequency_count(words_all_flatten)
    data = generate_data(keywords, words_all_flatten)
    data = np.array(data)
    return data

def get_model_x(input_shape1,input_shape2,output_shape=3):

    in_tensor_txt = tf.keras.layers.Input(shape=input_shape1)
    x = tf.keras.layers.Dense(128, activation='sigmoid')(in_tensor_txt)
    txt_feature = tf.keras.layers.Dense(64, activation='sigmoid')(x)

    in_tensor_img = tf.keras.layers.Input(shape=input_shape2)
    x = tf.keras.layers.Conv2D(32, (3, 3), activation='relu')(in_tensor_img)
    x = tf.keras.layers.MaxPooling2D((2, 2))(x)
    x = tf.keras.layers.Conv2D(64, (3, 3), activation='relu')(x)
    x = tf.keras.layers.MaxPooling2D((2, 2))(x)
    x = tf.keras.layers.Flatten()(x)
    img_feature = tf.keras.layers.Dense(64, activation='sigmoid')(x)

    x = tf.keras.layers.concatenate([txt_feature, img_feature])
    x = tf.keras.layers.Dense(output_shape, activation='softmax')(x)
    model = tf.keras.Model(inputs=[in_tensor_txt,in_tensor_img], outputs=x)
    return model


def get_model_txt(input_shape1,output_shape):
    in_tensor_txt = tf.keras.layers.Input(shape=input_shape1)
    x = tf.keras.layers.Dense(128, activation='sigmoid')(in_tensor_txt)
    txt_feature = tf.keras.layers.Dense(64, activation='sigmoid')(x)
    x = tf.keras.layers.Dense(output_shape, activation='softmax')(txt_feature)
    model = tf.keras.Model(inputs=in_tensor_txt, outputs=x)
    return model
def convert_data_to_datasets(keywords,data,label_value):
    if isinstance(data,list):
        txt_data = data[0]
        img_data = data[1]
    elif isinstance(data,dict):
        txt_data = data
    logits_data = convert_data_to_logits(keywords,txt_data)
    label = np.ones(len(logits_data))*label_value
    return logits_data,label


if __name__ == '__main__':
    root = '/home/topnet/图片'
    folders = ['head2', 'tail', 'other']
    keywords = [
        ['发包', '承包', '甲方', '乙方', '买方', '卖方', '全称', '承租', '出租', '分包', '建设单位', '施工单位','转让方','受让方','转让人','受让人','出卖','买受'],
        ['(以下简称', '（以下简称'],
        ['第一部分', '第一条', '第一节', '第一章'],
        ['一、', '一。', '一．', '一.'],
        ['协议书', '项目合同书', '采购合同', '销售合同', '买卖合同', '工程合同', '施工合同'],
        ['概况', '概述'],
        ['(盖章', '(签字', '(公章', '(盖单位章','（盖章', '（签字', '（公章','（盖单位章', '盖章)', '签字)', '公章)','盖单位章)', '盖章）', '签字）', '公章）''盖单位章）','(签名','（签名','签名）','签名)','(印章','（印章','印章）','印章)',],
        ['附件','附录','目录','说明'],
        ['中华人民共和国'],
        ['有限公司'],
        ['代理','委托'],
        ['法定代表'],
        ['补充'],
        ['自愿','公平'],
        ['有限公司'],
    ]
    keywords_flatten = [keywords[i][j] for i in range(len(keywords)) for j in range(len(keywords[i]))]
    img_size=320
    output_shape = 3
    r = r"[|_|.|!|+|-|=|—|,|$|￥|%|^|，|。|？|、|~|@|#|￥|%|…|&|*|《|》|<|>|「|」|{|}|【|】|(|)|/|]|:|：|；|‘|’|“|”|,|（|）"
    ([head, tail, other],[head_img,tail_img,other_img]) = read_image_txt(root, folders,img_size,use_img=False)
    head_data,head_label = convert_data_to_datasets(keywords_flatten,head,1)
    tail_data,tail_label = convert_data_to_datasets(keywords_flatten,tail,2)
    other_data,other_label = convert_data_to_datasets(keywords_flatten,other,0)
    text_input = np.concatenate((head_data, tail_data,other_data), axis=0)
    img_input = np.concatenate((head_img, tail_img,other_img), axis=0)
    label = np.concatenate((head_label, tail_label,other_label), axis=0)
    # tran test split
    x1_train, x1_test, x2_train, x2_test, y_train_raw, y_test_raw = train_test_split(text_input,img_input, label, random_state=1, train_size=0.7)

    # tensorflow mix model
    # y_train = tf.one_hot(y_train_raw,output_shape)
    # y_test = tf.one_hot(y_test_raw,output_shape)

    # model = get_model_x(input_shape1=(len(keywords_flatten),),input_shape2=(img_size,img_size,3),output_shape)
    # model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    # model.fit([x1_train,x2_train], y_train, epochs=100, batch_size=32,validation_data=([x1_test,x2_test],y_test),callbacks=[tf.keras.callbacks.ModelCheckpoint('./contract_models/model.h5',save_best_only=True)])

    # model = get_model_txt(input_shape1=(len(keywords_flatten),),output_shape)
    # model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    # model.fit(x1_train, y_train, epochs=100, batch_size=32,validation_data=(x1_test,y_test),callbacks=[tf.keras.callbacks.ModelCheckpoint('./contract_models/model.h5',save_best_only=True)])

    clf = svm.SVC(C=0.8, kernel='linear', gamma=20, decision_function_shape='ovr',verbose=True)
    clf.fit(x1_train, y_train_raw)
    y_pred = clf.predict(x1_test)
    y_pred[y_pred == 2] = 0
    y_test_raw[y_test_raw == 2] = 0
    print(accuracy_score(y_test_raw, y_pred))

    with open('model.pkl', 'wb') as f:
        pickle.dump(clf, f)
