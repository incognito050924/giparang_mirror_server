import os
import cv2
import numpy as np
from scipy.misc import imread, imresize
import matplotlib.pyplot as plt
import skimage.io
import skimage.transform

def preprocess_input(x, v2=True):
    x = x.astype('float32')
    x = x / 255.0
    if v2:
        x = x - 0.5
        x = x * 2.0
    return x

def load_img(file_name, read_gray=False):
    return imread(name=file_name, flatten=read_gray)

def load_img_with_cv2(file_name, read_gray=False):
    if read_gray:
        return cv2.imread(file_name, cv2.IMREAD_GRAYSCALE)
    else:
        return cv2.imread(file_name)

def load_img_from_dir(dir_path):
    valid_exts = [".jpg", ".gif", ".png", ".tga", ".jpeg"]

    # Append Images and their Names to Lists
    imgs = []
    names = []
    for f in os.listdir(dir_path):
        # For all files
        ext = os.path.splitext(f)[-1]
        # Check types
        if ext.lower() not in valid_exts:
            continue
        fullpath = os.path.join(dir_path, f)
        imgs.append(imread(fullpath))
        names.append(os.path.splitext(f)[0] + os.path.splitext(f)[1])

    return imgs, names

def rgb2gray(img):
    if len(img.shape) is 3:
        return np.dot(img[...,:3], [0.299, 0.587, 0.114])
    else:
        print ("Current Image if GRAY!")
        return img

def rgb2bgr(img):
    bgr = img.copy()
    # Blue Channel
    bgr[:, :, 0] = img[:, :, 2]
    # Green Channel
    # bgr[:, :, 1] = img[:, :, 1]
    # Red Channel
    bgr[:, :, 2] = img[:, :, 0]

    return bgr

def bgr2rgb(img):
    rgb = img.copy()
    # Red Channel
    rgb[:, :, 0] = img[:, :, 2]
    # Green Channel
    # rgb[:, :, 1] = img[:, :, 1]
    # Blue Channel
    rgb[:, :, 2] = img[:, :, 0]

def print_type_and_shape(img):
    print("Type is %s" % (type(img)))
    print("Shape is %s" % (img.shape,))

def mat2vec(img):
    return np.reshape(img, (1, -1))

def vec2mat(img, img_shape):
    return np.reshape(img, img_shape)

def resize_img_with_cv2(img, new_shape):
    h, w = img.shape[:2]
    new_h, new_w = new_shape[:2]

    interpolation = cv2.INTER_LINEAR
    if new_h - h < 0 and new_w - w < 0:
        # 기존 이미지 크기 보다 작게 만듦
        interpolation = cv2.INTER_AREA

    return cv2.resize(img, dsize=(new_w, new_h), interpolation=interpolation)

def gray2bgr(img):
    return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

def resize_img(img, new_shape):
    return imresize(img, new_shape)

def plot_img(imgs):
    for i, curr_img in enumerate(imgs):
        plt.figure(i)
        plt.imshow(curr_img)
        plt.title("[" + str(i) + "] ")
        plt.draw()

def preprocess_input(x, v2=True):
    x = x.astype('float32')
    x = x / 255.0
    if v2:
        x = x - 0.5
        x = x * 2.0
    return x