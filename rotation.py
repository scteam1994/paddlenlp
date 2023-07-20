import cv2

def rotate(img,angle):
    h,w=img.shape[:2]
    center=(w//2,h//2)
    M=cv2.getRotationMatrix2D(center,angle,1)
    rotated=cv2.warpAffine(img,M,(w,h))
    return rotated