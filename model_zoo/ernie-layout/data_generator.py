import shutil

from paddlenlp.utils.image_utils import two_dimension_sort_layout, img2base64, Bbox, check
from datasets import Dataset, Features, ClassLabel, Image, Sequence, Value
import os
from sklearn.model_selection import train_test_split
from paddleocr import PaddleOCR
from functools import cmp_to_key
import random
import PIL
from tqdm import tqdm

workspace_path = 'E:\\'
label_dict = {
    "invoice": 0,
    "contract": 1,
    "bank_receipt": 2,
    "central_unified": 3,
    "image_progress": 4,
    "quantities_valuation": 5,
    "statement_accounting_balance": 6,
    "business_license": 7,
    "commercial_housing_sales_ledger": 8,
    "total_output_value_construction_schedule": 9,
    "project_approval_documents": 10,
    "document_filing": 11,
    "site_construction_photos": 12,
    "equipment_in_place_photo": 13,
    "project_site_verification_form": 14,
    "organization_code_certificate": 15,
    "public_institution_legal_person_certificate": 16,
    "construction_permit_for_construction_project": 17,
    "state_owned_land_use_right_certificate": 18,
    "construction_land_use_permit": 19,
    "other": 20,
}


class DataLoader:
    def __init__(self, max_key_len=16, max_seq_len=512, image_size=1024):
        self.ocr = PaddleOCR(use_angle_cls=True, show_log=False, use_gpu=True, lang="ch")
        self.image_size = image_size
        self.fields = label_names

    def get_ocr_res(self, image_path):  # 使用 paddleOCR 提取图片中的文字
        ocr_res = self.ocr.ocr(image_path)
        return ocr_res

    def ocr2feature(self, ocr_res, image_path, label):
        segments = []
        for rst in ocr_res:  # box 是有旋转角度的
            top_left = rst[0][0]
            top_right = rst[0][1]
            bottom_right = rst[0][2]
            bottom_left = rst[0][3]
            top_left_x = top_left[0]
            top_left_y = top_left[1]
            top_right_x = top_right[0]
            top_right_y = top_right[1]
            bottom_right_x = bottom_right[0]
            bottom_right_y = bottom_right[1]
            bottom_left_x = bottom_left[0]
            bottom_left_y = bottom_left[1]
            left = min(top_left_x, bottom_left_x)
            top = min(top_left_y, top_right_y)
            right = max(top_right_x, bottom_right_x)
            bottom = max(bottom_left_y, bottom_right_y)
            width = right - left
            height = bottom - top
            text = rst[1][0]
            segments.append({
                "bbox": Bbox(*[left, top, width, height]),
                "text": text
            })
        segments.sort(key=cmp_to_key(two_dimension_sort_layout))  # 对 bbox 排序
        img_base64 = img2base64(image_path)
        if len(segments) == 0:  # 如果图片中没有文字，OCR 返回的结果为空，则 segments 为空，此时 im_w_box = 0, im_h_box = 0
            im_w_box = 0
            im_h_box = 0
        else:
            im_w_box = max([seg["bbox"].left + seg["bbox"].width for seg in segments]) + 20  # 通过box找最大的 right
            im_h_box = max([seg["bbox"].top + seg["bbox"].height for seg in segments]) + 20  # 通过box找最大的 bottom
        img = PIL.Image.open(image_path)
        im_w, im_h = img.size  # 图片本身的宽和高
        width = im_w
        height = im_h
        im_w, im_h = max(im_w, im_w_box), max(im_h, im_h_box)
        texts = []
        bboxes = []
        segment_bboxes = []
        segment_ids = []
        seg_tokens = []
        for segment_id, segment in enumerate(segments):  # 分词，如果是中文，将每个中文汉字字符切分；如果是英文或数字，则不切分
            bbox = segment["bbox"]
            bbox_left, bbox_top, bbox_width, bbox_height = bbox.left, bbox.top, bbox.width, bbox.height
            if bbox_width < 0: raise ValueError("Incorrect bbox, please check the input word boxes. ")
            text = segment["text"]
            texts.append(text)
            bboxes.append([int(bbox.left), int(bbox.top), int(bbox.right), int(bbox.bottom)])
            char_num = []
            eng_word = ""
            for char in text:
                if not check(char) and not eng_word:  # 检查是否为英文字母或数字，判断是否需要分词
                    seg_tokens.append(char)
                    segment_ids.append(segment_id)
                    char_num.append(2)
                elif not check(char) and eng_word:
                    seg_tokens.append(eng_word)
                    segment_ids.append(segment_id)
                    char_num.append(len(eng_word))
                    eng_word = ""
                    seg_tokens.append(char)
                    segment_ids.append(segment_id)
                    char_num.append(2)
                else:
                    eng_word += char
            if eng_word:
                seg_tokens.append(eng_word)
                segment_ids.append(segment_id)
                char_num.append(len(eng_word))
            ori_char_width = round(bbox_width / sum(char_num), 1)  # 用 bbox 的宽度除以改bbox中字符的数量得到每个字符的宽度
            for chr_idx in range(len(char_num)):  # 根据字符的宽度得到字符的bbox，保存在 segment_bbox中
                if chr_idx == 0:
                    seg_box_left = int(bbox_left)
                    seg_box_top = int(bbox_top)
                    seg_box_right = int(bbox_left + ori_char_width * char_num[chr_idx])
                    seg_box_bottom = int(bbox_top + bbox_height)
                    segment_bboxes.append([seg_box_left, seg_box_top, seg_box_right, seg_box_bottom])
                else:
                    seg_box_left = int(bbox_left + (ori_char_width * sum(char_num[:chr_idx])))
                    seg_box_top = int(bbox_top)
                    seg_box_right = int(
                        bbox_left + (ori_char_width * sum(char_num[:chr_idx])) + ori_char_width * char_num[chr_idx])
                    seg_box_bottom = int(bbox_top + bbox_height)
                    segment_bboxes.append([seg_box_left, seg_box_top, seg_box_right, seg_box_bottom])
        feature = {
            "name": image_path,
            "page_no": 0,
            "text": texts,
            "bbox": bboxes,
            "segment_bbox": segment_bboxes,
            "segment_id": segment_ids,
            "segment_text": seg_tokens,
            "image": img_base64,
            "width": int(width),
            "height": int(height),
            "md5sum": "",
            "qas": self.gen_qas(label)
        }

        return feature

    def gen_qas(self, label: str):
        qas = {
            'question_id': [-1],
            'question': ['What is the document type?'],
            'answers': [
                {
                    'text': [label, ],
                    'answer_start': [-1],
                    'answer_end': [-1]
                }
            ]
        }
        return qas

def get_dict_key_by_value(dict, value):
    for k, v in dict.items():
        if v == value:
            return k
    return None
def split_by_label(root, label_path):
    with open(label_path, 'r') as f:
        label_list = f.readlines()
    for label in label_list:
        file_label = label.split(' ')
        if len(file_label) != 2:
            print('label error: ', label)
            continue
        ori_filepath = os.path.join(root, file_label[0])
        if not os.path.exists(ori_filepath):
            print('file not exist: ', ori_filepath)
            continue
        target_dir = os.path.join(os.path.join(workspace_path, "data_sample"), get_dict_key_by_value(label_dict, int(file_label[1])))
        # copyfrom ori_filepath to target_dir
        shutil.copy(ori_filepath, target_dir)


def mkdir(path):
    if not os.path.exists(path):
        os.mkdir(path)


if __name__ == '__main__':
    # for k in label_dict.keys():
    #     try:
    #         mkdir(os.path.join(os.path.join(workspace_path, "data_sample"), k))
    #     except:
    #         pass
    # split_by_label(workspace_path, os.path.join(workspace_path, 'train_list.txt'))
    dataset_dir = os.path.join(workspace_path, "data_sample")
    record_lines = []
    label_names = list(label_dict.keys())
    # label_names = ['central_unified']
    label_count = [0 for _ in range(len(label_names))]
    image_list = []
    label_list = []
    # label_a_dir_path = os.path.join(dataset_dir, "label_A")
    # label_b_dir_path = os.path.join(dataset_dir, "label_B")
    # label_a_images = os.listdir(label_a_dir_path)
    # label_b_images = os.listdir(label_b_dir_path)
    # for label_a_image_name in label_a_images:
    #     ext = label_a_image_name.split(".")[-1]
    #     if ext == "png":
    #         image_list.append(os.path.join(label_a_dir_path, label_a_image_name))
    #         label_list.append("label_A")
    # for label_b_image_name in label_b_images:
    #     ext = label_b_image_name.split(".")[-1]
    #     if ext == "png":
    #         image_list.append(os.path.join(label_b_dir_path, label_b_image_name))
    #         label_list.append("label_B")

    for l in label_names:
        label_dir_path = os.path.join(dataset_dir, l)
        label_images = os.listdir(label_dir_path)
        for label_a_image_name in label_images:
            ext = label_a_image_name.split(".")[-1]
            if ext in ["png","jpg","jpeg"]:
                image_list.append(os.path.join(label_dir_path, label_a_image_name))
                label_list.append(l)
    features = []
    label_ids = []
    ziplist = list(zip(image_list, label_list))
    random.shuffle(ziplist)
    image_loader = DataLoader()
    for image_path, label in tqdm(ziplist):
        try:
            ocr_res = image_loader.get_ocr_res(image_path)
            feature = image_loader.ocr2feature(ocr_res[0], image_path, label)
            features.append(feature)
            label_ids.append(label_list.index(label))
        except Exception as err:
            print(f"file: {image_path}, label: {label},  err: {err}")
    train_features, validation_features, train_labels, validation_labels = train_test_split(features, label_ids,
                                                                                            test_size=0.3,
                                                                                            random_state=1000,
                                                                                            shuffle=True)
    train_ds = Dataset.from_list(train_features)
    eval_ds = Dataset.from_list(validation_features)
    train_ds_save_path = os.path.join(workspace_path, "train_sample_v1")
    train_ds.save_to_disk(train_ds_save_path)
    eval_ds_save_path = os.path.join(workspace_path, "eval_sample_v1")
    eval_ds.save_to_disk(eval_ds_save_path)
