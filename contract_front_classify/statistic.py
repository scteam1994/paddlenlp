import json
import os
import numpy as np
import cv2
from tqdm import tqdm
import pickle
import jieba

from contract_front_classify.parameter import Parameter


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
                try:
                    img = normalize(img, img_size)
                except:
                    print()
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
                            contains[i] = word.count(keyword)
                    elif isinstance(keyword, list):
                        for kw in keyword:
                            if kw in word:
                                contains[i] = word.count(keyword)
        elif isinstance(head_words, str):
            for i, keyword in enumerate(keywords):
                if isinstance(keyword, str):
                    if keyword in head_words:
                        contains[i] = head_words.count(keyword)
                elif isinstance(keyword, list):
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
                        for kw in keyword[1:]:
                            if kw in head_words:
                                contains[i] = head_words.count(kw)
                                loc.append(head_words.find(kw) / len(head_words))
                            else:
                                loc.append(-1)
        contains_data.append(np.hstack((contains, np.array(loc))))
    return contains_data


def convert_data_to_logits(keywords, data):
    words_all = list(data.values())
    words_all_flatten = [flatten_txt(words_all[i]) for i in range(len(words_all))]
    # words_frq = frequency_count(words_all_flatten)
    data = generate_data(keywords, words_all_flatten)
    data = np.array(data)
    return data




def convert_data_to_datasets(keywords, data, label_value):
    if isinstance(data, list):
        txt_data = data[0]
        img_data = data[1]
    elif isinstance(data, dict):
        txt_data = data
    logits_data = convert_data_to_logits(keywords, txt_data)
    label = np.ones(len(logits_data)) * label_value
    return logits_data, label



def save_variable(data,name):

    # if isinstance(data,dict):
    if isinstance(data,np.ndarray):
        np.save(f'contract_front_classify/{name}.npy',data)
    else:
        with open(f'contract_front_classify/{name}.pkl','wb') as f:
            pickle.dump(data,f)



if __name__ == '__main__':
    param = Parameter()
    img_size = param.img_size
    output_shape = param.output_shape
    tensorflow_backend = param.tensorflow_backend
    svm_backend = param.svm_backend
    faltten = param.flatten
    use_img = param.use_img
    show_res = param.show_res
    root = param.root
    folders = param.folders

    keywords_flatten = []
    for i in param.keywords:
        word_limit = i[0]
        for j in word_limit:
            for k in i[1:]:
                if isinstance(k, str):
                    keywords_flatten.append([j, k])
                else:
                    print()
    key_words_simple = []
    for i in param.keywords:
        word_limit = i[0]
        for j in word_limit:
            d = i[1:]
            d.insert(0, j)
            key_words_simple.append(d)

    if faltten:
        param.keywords = keywords_flatten
    r = r"[|_|.|!|+|-|=|—|,|$|￥|%|^|，|。|？|、|~|@|#|￥|%|…|&|*|《|》|<|>|「|」|{|}|【|】|(|)|/|]|:|：|；|‘|’|“|”|,|（|）"
    ([head, tail, other], [head_img, tail_img, other_img]) = read_image_txt(root, folders, img_size, use_img=use_img)
    head_data, head_label = convert_data_to_datasets(param.keywords, head, 1)
    tail_data, tail_label = convert_data_to_datasets(param.keywords, tail, 2)
    other_data, other_label = convert_data_to_datasets(param.keywords, other, 0)
    text_input = np.concatenate((head_data, tail_data, other_data), axis=0)
    img_input = np.concatenate((head_img, tail_img, other_img), axis=0)
    label = np.concatenate((head_label, tail_label, other_label), axis=0)
    img_name = list(head.keys()) + list(tail.keys()) + list(other.keys())
    save_variable(img_name,name = 'img_name')
    save_variable(text_input,name ='text_input')
    save_variable(img_input,name = 'img_input')
    save_variable(label,name = 'label')
