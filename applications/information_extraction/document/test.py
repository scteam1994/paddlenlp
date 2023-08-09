from time import time

from paddleocr import PaddleOCR
from paddlenlp import Taskflow
import cv2

# ocr_version对应模型设置
# ocr = PaddleOCR(use_angle_cls=True, lang="ch", ocr_version="PP-OCRv3", det_db_box_thresh=0.1, use_dilation=True)

# def do_ocr():
#     ocr识别
#     ocr_result = ocr.ocr("./images/image16.png", rec=False)
#     draw result
#     image = cv2.imread("C:\Users\topnet\Desktop\invoice/image16.png")
#     for line in ocr_result:
#         print(line)
#         points = line[0]
#         text = line[1][0]
#         cv2.polylines(image, [points], True, (0, 255, 0), 2)
#         cv2.putText(image, text, (points[0][0], points[0][1]), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
#     cv2.imwrite("./images/image16_result.png", image)
#     ocr结果组成layout参数pip
#     ocr_layout = []
#     for res in ocr_result:
#         for item in res:
#             x1, y1 = item[0][0]
#             x2, y2 = item[0][2]
#             text = item[1][0]
#             ocr_layout.append(([x1, y1, x2, y2], text))
#     return ocr_layout

ie_task = Taskflow("information_extraction", schema=["开户行及账号"], model="uie-x-base", layout_analysis=False,
                   max_seq_len=1024,position_prob=0.3)

# uie模型预测
# ocr_layout = do_ocr()
# ie_result = ie_task({"doc": "./images/image16.png", "layout": ocr_layout})

# print(ie_result)
t = time()
for _ in range(1):
    ie_result2 = ie_task({"doc": "./images/image16.png"})
print(time()-t)
print(ie_result2)
