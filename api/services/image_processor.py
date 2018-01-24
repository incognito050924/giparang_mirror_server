import cv2
import numpy as np

def bytes2opencv_img(img_raw):
    img_str = img_raw.read()
    img_nparr = np.fromstring(img_str, np.uint8)
    img = cv2.imdecode(img_nparr, cv2.IMREAD_COLOR)
    # cv2.imshow('Uploaded', img)
    # cv2.waitKey()
    # cv2.destroyAllWindows()
    return img

def bgr2rgb(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

def resize_image(img, new_shape):
    h, w = img.shape[:2]
    new_h, new_w = new_shape[:2]

    interpolation = cv2.INTER_LINEAR
    if new_h - h < 0 and new_w - w < 0:
        # 기존 이미지 크기 보다 작게 만듦
        interpolation = cv2.INTER_AREA

    return cv2.resize(img, dsize=(new_w, new_h), interpolation=interpolation)
