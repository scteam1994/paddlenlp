import glob
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

import jieba


def frequency_count(txt_list):
    counts = {}
    position = {}
    all_cnt = 0
    for txt in txt_list:
        all_cnt += len(txt)
        words = jieba.lcut(txt)

        for word in words:
            if len(word) == 1:
                continue
            else:
                counts[word] = counts.get(word, 0) + 1
                if word not in position.keys():
                    position[word] = [txt.find(word)]
                else:
                    position[word].append(txt.find(word))
    counts = {k: v for k, v in counts.items()}
    res = sorted(position.items(), key=lambda x: len(x[1]), reverse=True)
    res2 = sorted(position.items(), key=lambda x: np.std(x[1]), reverse=True)
    sta = {}
    for r in res2:
        if len(r[1]) > 50:
            sta[r[0]] = [np.mean(r[1]), np.std(r[1]), len(r[1])]

    return sta


def extract_text(root):
    import paddle
    from paddleocr import PaddleOCR
    ocr = PaddleOCR(use_angle_cls=True, lang="ch", ocr_version="PP-OCRv4")
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


def read_image_txt(root, folders, img_size, use_img=True):
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
            imgs = [np.zeros((1))] * len(data)
        else:
            for img_path in data.keys():
                img = cv2.imread(os.path.join(root, folder, img_path))
                img = normalize(img, img_size)
                imgs.append(img)
        img_datas.append(imgs)
    return (datas, img_datas)


def normalize(img, img_size):
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


def flatten_txt(txt_list: list):
    txt = ''
    for line in txt_list:
        if isinstance(line, list):
            txt += flatten_txt(line)
        elif isinstance(line, str):
            txt += line
    return txt


def generate_data(keywords, words_all):
    contains_data = []
    location = []
    for head_words in words_all:
        contains = np.zeros(len(keywords))
        loc = []
        if isinstance(head_words, list):
            for word in head_words:
                for i, keyword in enumerate(keywords):
                    if isinstance(keyword, str):
                        if keyword in word:
                            contains[i] = 1
                    elif isinstance(keyword, list):
                        for kw in keyword:
                            if kw in word:
                                contains[i] = 1
        elif isinstance(head_words, str):
            for i, keyword in enumerate(keywords):
                if isinstance(keyword, str):
                    if keyword in head_words:
                        contains[i] = 1
                elif isinstance(keyword, list):
                    # for kw in keyword:
                    #     if kw in verf_data:
                    #         contains[i] = 1
                    if "all" == keyword[0]:
                        verf_data = head_words
                        for kw in keyword[1:]:
                            if kw in verf_data:
                                contains[i] = 1
                                loc.append(head_words.find(kw)/len(head_words))
                            else:
                                loc.append(-1)
                    elif type(keyword[0]) == int:
                        verf_data = head_words[:keyword[0]]
                        for kw in keyword[1:]:
                            if kw in verf_data:
                                contains[i] = 1
                    else:
                        for kw in keyword:
                            if kw in head_words:
                                contains[i] = 1
        contains_data.append(np.hstack((contains,np.array(loc))))
    return contains_data


def convert_data_to_logits(keywords, data):
    words_all = list(data.values())
    words_all_flatten = [flatten_txt(words_all[i]) for i in range(len(words_all))]
    words_frq = frequency_count(words_all_flatten)
    data = generate_data(keywords, words_all_flatten)
    data = np.array(data)
    return data


def get_model_x(input_shape1, input_shape2, output_shape=3):
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
    model = tf.keras.Model(inputs=[in_tensor_txt, in_tensor_img], outputs=x)
    return model


def get_model_txt(input_shape1, output_shape):
    in_tensor_txt = tf.keras.layers.Input(shape=input_shape1)
    # x = tf.keras.layers.Conv1D(32, 15, activation='relu',padding="same")(in_tensor_txt)
    # x = tf.keras.layers.Conv1D(32, 3, activation='relu',padding="same")(x)
    x = tf.keras.layers.Flatten()(in_tensor_txt)
    # x = tf.keras.layers.Dense(256, activation='sigmoid')(x)
    x = tf.keras.layers.Dense(64, activation='relu')(x)
    x = tf.keras.layers.Dense(output_shape, activation='softmax')(x)
    model = tf.keras.Model(inputs=in_tensor_txt, outputs=x)
    return model


def convert_data_to_datasets(keywords, data, label_value):
    if isinstance(data, list):
        txt_data = data[0]
        img_data = data[1]
    elif isinstance(data, dict):
        txt_data = data
    logits_data = convert_data_to_logits(keywords, txt_data)
    label = np.ones(len(logits_data)) * label_value
    return logits_data, label


if __name__ == '__main__':
    root = '/home/topnet/图片'
    folders = ['head2', 'tail', 'other']
    keywords = [
        [[20, 50, 100,'all'], '销售合同', '供货合同', '购销合同', '承揽合同', '销购合同', '采购协议', '供应合同',
         '定做合同', '加工合同'],
        [[20, 70, 'all'], '协议'],
        [[20, 80,'all'], '项目合同书', '采购合同', '销售合同', '买卖合同', '工程合同', '施工合同', '合同条款',
         '分包合同', '安装合同', '服务合同', '维护合同', '承包合同'],
        [['all'],'甲方', '乙方','买方', '卖方',],
        [[50,200], '发包', '承包', '全称', '承租', '出租', '分包', '建设单位',
         '施工单位', '转让方', '受让方', '转让人', '受让人', '出卖', '买受', '供方', '需方', '委托人', '代建人',
         '委托方', '代建方'],
        [[50, 200], '编号', '地点', '时间', '签订'],
        [['all'], '(以下简称', '（以下简称'],
        [[50,'all'], '第一', '第一部分', '第一条', '第一节', '第一章'],
        [['all'], '一、', '一。', '一．', '一.'],

        [['all'], '概况', '概述'],
        [['all'], '(盖章', '(签字', '(公章', '(盖单位章', '（盖章', '（签字', '（公章', '（盖单位章', '盖章)', '签字)',
         '公章)', '盖单位章)', '盖章）', '签字）', '公章）''盖单位章）', '(签名', '（签名', '签名）', '签名)', '(印章',
         '（印章', '印章）', '印章)', ],
        [[10, 20, 'all'], '附件', '附录', '目录', '说明'],
        [['all'], '中华人民共和国'],
        [['all'], '有限公司'],
        [['all'], '代理', '委托'],
        [['all'], '法定代表'],
        [['all'], '补充'],
        [['all'], '自愿', '公平'],
        [['all'], '有限公司'],
    ]
    keywords = [
        [['all'],'销售合同','供货合同','购销合同','承揽合同','销购合同','采购协议','供应合同','定做合同','加工合同'],
        [['all'],'甲方', '乙方', '买方', '卖方'],
        [['all'],'发包', '承包', '全称', '承租', '出租', '分包', '建设单位', '施工单位','转让方','受让方','转让人','受让人','出卖','买受','供方','需方','委托人','代建人','委托方','代建方'],
        [['all'], '编号','地点','时间','签订'],
        [['all'],'(以下简称', '（以下简称'],
        [['all'],'第一部分', '第一条', '第一节', '第一章'],
        [['all'],'一、', '一。', '一．', '一.'],
        [['all'], '协议'],
        [['all'], '项目合同书', '采购合同', '销售合同', '买卖合同', '工程合同', '施工合同','合同条款','分包合同','安装合同','服务合同','维护合同','承包合同'],
        [['all'],'概况', '概述'],
        [['all'],'(盖章', '(签字', '(公章', '(盖单位章','（盖章', '（签字', '（公章','（盖单位章', '盖章)', '签字)', '公章)','盖单位章)', '盖章）', '签字）', '公章）''盖单位章）','(签名','（签名','签名）','签名)','(印章','（印章','印章）','印章)',],
        [['all'],'附件','附录','目录','说明'],
        [['all'],'中华人民共和国'],
        [['all'],'有限公司'],
        [['all'],'代理','委托'],
        [['all'],'法定代表'],
        [['all'],'补充'],
        [['all'],'自愿','公平'],
        [['all'],'有限公司'],
    ]
    keywords_flatten = []
    for i in keywords:
        word_limit = i[0]
        for j in word_limit:
            for k in i[1:]:
                if isinstance(k, str):
                    keywords_flatten.append([j, k])
                else:
                    print()
    key_words_simple = []
    for i in keywords:
        word_limit = i[0]
        for j in word_limit:
            d = i[1:]
            d.insert(0, j)
            key_words_simple.append(d)

    img_size = 320
    output_shape = 3
    tensorflow_backend = False
    svm_backend = True
    r = r"[|_|.|!|+|-|=|—|,|$|￥|%|^|，|。|？|、|~|@|#|￥|%|…|&|*|《|》|<|>|「|」|{|}|【|】|(|)|/|]|:|：|；|‘|’|“|”|,|（|）"
    ([head, tail, other], [head_img, tail_img, other_img]) = read_image_txt(root, folders, img_size, use_img=False)
    head_data, head_label = convert_data_to_datasets(keywords_flatten, head, 1)
    tail_data, tail_label = convert_data_to_datasets(keywords_flatten, tail, 0)
    other_data, other_label = convert_data_to_datasets(keywords_flatten, other, 0)
    text_input = np.concatenate((head_data, tail_data, other_data), axis=0)
    img_input = np.concatenate((head_img, tail_img, other_img), axis=0)
    label = np.concatenate((head_label, tail_label, other_label), axis=0)
    img_name = list(head.keys()) + list(tail.keys()) + list(other.keys())
    # tran test split
    x1_train, x1_test, x2_train, x2_test, y_train_raw, y_test_raw, img_name_train, img_name_test = train_test_split(text_input, img_input, label,img_name,
                                                                                     random_state=113, train_size=0.7)
    if tensorflow_backend:
        # tensorflow mix model
        import tensorflow as tf
        y_train = tf.one_hot(y_train_raw,output_shape)
        y_test = tf.one_hot(y_test_raw,output_shape)

        # model = get_model_x(input_shape1=(len(keywords_flatten),),input_shape2=(img_size,img_size,3),output_shape)
        # model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        # model.fit([x1_train,x2_train], y_train, epochs=100, batch_size=32,validation_data=([x1_test,x2_test],y_test),callbacks=[tf.keras.callbacks.ModelCheckpoint('./contract_models/model.h5',save_best_only=True)])

        model = get_model_txt(input_shape1= (x1_train.shape[1],1)
                              ,output_shape=output_shape)
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        model.summary()
        model.fit(x1_train[:,:,np.newaxis], y_train, epochs=100, batch_size=32,validation_data=(x1_test[:,:,np.newaxis],y_test),callbacks=[tf.keras.callbacks.ModelCheckpoint('./contract_models/model.h5',save_best_only=True)])

    # 'sigmoid', 'poly', 'precomputed', 'rbf', 'linear'
    # 'ovr', 'ovo'
    if svm_backend:
        clf = svm.SVC(C=0.85, kernel='linear', decision_function_shape='ovr', verbose=True)
        clf.fit(x1_train, y_train_raw)
        y_pred = clf.predict(x1_test)
        y_pred[y_pred == 2] = 0
        y_test_raw[y_test_raw == 2] = 0
        y_pred_train = clf.predict(x1_train)
        y_pred_train[y_pred_train == 2] = 0
        y_train_raw[y_train_raw == 2] = 0

        for i in np.where(y_test_raw != y_pred)[0]:
            print(img_name_test[i])
            img_path = glob.glob(os.path.join(root, '*', img_name_test[i]))[0]
            img = cv2.imread(img_path)
            label = img_path.split('/')[-2]
            # cv2.imshow(f'{label}', img)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()

        print('-------------------')
        for i in np.where(y_train_raw != y_pred_train)[0]:
            print(img_name_train[i])
            img_path = glob.glob(os.path.join(root, '*', img_name_train[i]))[0]
            img = cv2.imread(img_path)
            label = img_path.split('/')[-2]
            # cv2.imshow(f'{label}', img)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()
        print('all test num : ', len(y_test_raw))
        print(accuracy_score(y_test_raw, y_pred))

        print(accuracy_score(y_train_raw, y_pred_train))
        with open('contract_front_classify/model_loc.pkl', 'wb') as f:
            pickle.dump(clf, f)
