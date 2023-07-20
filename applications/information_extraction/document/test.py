
from paddleocr import PaddleOCR
from paddlenlp import Taskflow
import cv2
# ocr_version对应模型设置
ocr = PaddleOCR(use_angle_cls=True, lang="ch", ocr_version="PP-OCRv3", det_db_box_thresh=0.1, use_dilation=True)

# ocr识别
ocr_result = ocr.ocr("./images/14.png", rec=True)
#draw result
image = cv2.imread("./images/14.png")
for line in ocr_result:
    print(line)
    points = line[0]
    text = line[1][0]
    cv2.polylines(image, [points], True, (0, 255, 0), 2)
    cv2.putText(image, text, (points[0][0], points[0][1]), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
cv2.imwrite("./images/14_result.png", image)
# ocr结果组成layout参数pip
ocr_layout = []
for res in ocr_result:
    for item in res:
        x1, y1 = item[0][0]
        x2, y2 = item[0][2]
        text = item[1][0]
        ocr_layout.append(([x1, y1, x2, y2], text))

ie_task = Taskflow("information_extraction", schema=["开户行及账号"], model="uie-x-base", layout_analysis=True)

# uie模型预测
ie_result = ie_task({"doc": "./images/14.png", "layout": ocr_layout})

print(ie_result)

ie_result2 = ie_task({"doc": "./images/14.png"})
print(ie_result2)