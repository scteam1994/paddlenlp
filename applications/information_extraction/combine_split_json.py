import os
import json
import shutil

import PIL.Image
import numpy as np
import cv2


def get_point(event, x, y, flags, param):
    # 鼠标单击事件
    if event == cv2.EVENT_LBUTTONDOWN:
        # 输出坐标
        print('坐标值: ', x, y)
        # 在传入参数图像上画出该点
        # cv2.circle(param, (x, y), 1, (255, 255, 255), thickness=-1)
        img = param.copy()
        # 输出坐标点的像素值
        print('像素值：', param[y][x])  # 注意此处反转，(纵，横，通道)
        # 显示坐标与像素
        text = "(" + str(x) + ',' + str(y) + ')' + str(param[y][x])
        cv2.putText(img, text, (0, param.shape[0]), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 0, 255), 1)
        cv2.imshow('img', img)
    if event == cv2.EVENT_RBUTTONDOWN:
        # save imge
        cv2.imwrite('save.jpg', param)
        # cv2.waitKey(0)


def show_img(image):
    cv2.namedWindow("img", cv2.WINDOW_NORMAL)
    cv2.setMouseCallback('img', get_point, image)
    cv2.resizeWindow("img", 1000, 600)
    # while True:
    cv2.imshow('img', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def save_img(img, save_path):
    cv2.imwrite(save_path, img)


def combine_json(json_list):
    json_combine = []
    img_list_combine = []
    file_cnt = 0
    for json_file in json_list:
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

            img_list, data, file_cnt = rename_img_and_copy(data, file_cnt, json_file)
            json_combine.extend(data)
            img_list_combine.extend(img_list)
    return img_list_combine, json_combine


def rename_img_and_copy(json_combine, cnt, json_file=None):
    img_list = []
    img_list_raw = []
    img_rename_log = {}
    for i, json_data in enumerate(json_combine):

        img = json_data['data']['image']
        img_list_raw.append(img.split('-')[-1])
        ori_img_name = os.path.basename(img).split('-')[-1]
        tar_img_name = str(cnt) + '.' + ori_img_name.split('.')[-1]
        json_combine[i]['data']['image'] = img.replace(ori_img_name, tar_img_name)
        img_list.append(json_combine[i]['data']['image'])
        ori_img_path = os.path.join(os.path.dirname(json_file), 'images', ori_img_name)
        tar_img_path = os.path.join(os.path.join(OUT_dir, 'uie_data/images'), tar_img_name)
        img_rename_log[ori_img_path] = tar_img_path
        # cnt += 1
        # continue
        try:
            os.makedirs(os.path.dirname(tar_img_path))
        except:
            pass
        if os.path.exists(tar_img_path):
            print(f'file exists: {tar_img_path}')
            cnt += 1
            continue
        if os.path.exists(ori_img_path):
            # copy file from ori to tar
            shutil.copy(ori_img_path, tar_img_path)
        else:
            print(f'not exists: {ori_img_name}')
        cnt += 1
    with open(os.path.join(OUT_dir, f'uie_data/img_rename_log_{os.path.basename(json_file)}'), 'w',
              encoding='utf-8') as f:
        json.dump(img_rename_log, f, ensure_ascii=False)
    return img_list, json_combine, cnt


def split_json(json_comb, ratio=0.7, img_list=None):
    if img_list:
        img_train = np.random.choice(img_list, int(len(img_list) * ratio), replace=False)
        img_dev = list(set(img_list) - set(img_train))
        with open(os.path.join(OUT_dir, 'uie_data/train_list.txt'), 'w', encoding='utf-8') as f:
            f.write('\n'.join(img_train))
        with open(os.path.join(OUT_dir, 'uie_data/dev_list.txt'), 'w', encoding='utf-8') as f:
            f.write('\n'.join(img_dev))

    else:
        with open(os.path.join(OUT_dir, 'uie_data/train_list.txt'), 'r', encoding='utf-8') as f:
            img_train = f.read().split('\n')
        with open(os.path.join(OUT_dir, 'uie_data/dev_list.txt'), 'r', encoding='utf-8') as f:
            img_dev = f.read().split('\n')
    json_train = [json_data for json_data in json_comb if json_data['data']['image'] in img_train]
    json_dev = [json_data for json_data in json_comb if json_data['data']['image'] in img_dev]
    with open(os.path.join(OUT_dir, 'uie_data/train.json'), 'w', encoding='utf-8') as f:
        json.dump(json_train, f, ensure_ascii=False)
    with open(os.path.join(OUT_dir, 'uie_data/dev.json'), 'w', encoding='utf-8') as f:
        json.dump(json_dev, f, ensure_ascii=False)
    with open(os.path.join(OUT_dir, 'uie_data/train_s.json'), 'w', encoding='utf-8') as f:
        json.dump(json_train[:200], f, ensure_ascii=False)
    with open(os.path.join(OUT_dir, 'uie_data/dev_s.json'), 'w', encoding='utf-8') as f:
        json.dump(json_dev[:100], f, ensure_ascii=False)



def check_deprecated(inputs):
    outputs = []
    check_res = True
    for res in inputs:
        if 'rectanglelabels' not in res['value'].keys():
            print('no rectanglelabels')
            continue
        elif res['value']['rectanglelabels'][0] in ['名称', '开户行及账号', "纳税人识别号", "地址、电话", "价税合计",
                                                    "表头", "校验码"]:
            check_res = False
        else:
            outputs.append(res)
    return check_res, outputs


def check_and_combine_json(json_list):
    target_data = []
    image_names = []
    for json_file in json_list:
        img_root = os.path.join(os.path.dirname(json_file), 'images')

        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            for json_data in data:
                img_name = json_data['data']['image'].split('-')[-1]
                if json_data['annotations'][0]['result'] == []:
                    print(f'empty: {img_name}')
                    continue
                if 'invoice' in json_file:
                    check, json_data['annotations'][0]['result'] = check_deprecated(json_data['annotations'][0]['result'])
                    if not check:
                        print(f'deprecated: {img_name}')
                        continue

                if img_name in image_names:
                    print(f'duplicate: {img_name}')
                    continue
                if not os.path.exists(os.path.join(img_root, img_name)):
                    print(f'not exists: {img_name}')
                    continue
                try:
                    img = PIL.Image.open(os.path.join(img_root, img_name))
                    h ,w = img.height, img.width
                    if (json_data['annotations'][0]['result'][0]['original_height'], json_data['annotations'][0]['result'][0]['original_width']) != (h, w):
                        print(f'size mismatch: {img_name}')
                        continue

                except:
                    print(f'can not open: {img_name}')
                    continue
                target_data.append(json_data)
                image_names.append(img_name)
    with open(os.path.join(os.path.dirname(json_list[0]), 'checked.json'), 'w', encoding='utf-8') as f:
        json.dump(target_data, f, ensure_ascii=False)

        # img = cv2.imread(os.path.join(img_root, img_name))
        # img_w, img_h = img.shape[1], img.shape[0]
        # for e in json_data['annotations'][0]['result']:
        #     # box 取值方法参见paddlenlp/paddlenlp/utils/tools.py ：line 406
        #     box = [
        #         int(e["value"]["x"] * 0.01 * img_w),
        #         int(e["value"]["y"] * 0.01 * img_h),
        #         int((e["value"]["x"] + e["value"]["width"]) * 0.01 * img_w),
        #         int((e["value"]["y"] + e["value"]["height"]) * 0.01 * img_h),
        #     ]
        #
        #     cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]), (0, 0, 255), 2)
        # show_img(img)
Field = 'rest'
OUT_dir = f'/home/topnet/uie_data/{Field}'

if __name__ == '__main__':
    if Field == 'invoice':
        json_lists = [
            '/home/topnet/0714/invoice/project-77-at-2023-07-25-11-08-b751164d.json',
            '/home/topnet/0714/invoice/project-78-at-2023-07-24-13-02-89da60e7.json',
            '/home/topnet/0714/invoice/project-79-at-2023-07-25-09-23-6b7314b8.json',
            '/home/topnet/0714/invoice/project-80-at-2023-07-25-06-18-b34e09b1.json',
            '/home/topnet/0714/invoice/project-81-at-2023-07-25-12-40-fa44385f.json',
            '/home/topnet/0714/invoice/project-82-at-2023-07-24-13-01-416a98d1.json',
        ]
    else:
        json_lists = [
            '/home/topnet/0714/rest/rest.json',
        ]
    check_and_combine_json(json_lists)
    json_lists = [
        f'/home/topnet/0714/{Field}/checked.json',
    ]
    ratio = 0.7
    img_list, json_combine = combine_json(json_lists)
    split_json(json_combine, ratio, img_list)
