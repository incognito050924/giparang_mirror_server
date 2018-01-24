import os
import random
import numpy as np
import cv2
import dlib
import math
from .network import client

def gradation_width(img, from_color, to_color):
    h, w = img.shape[:2]
    if w <= 0:
        color_weight = np.array([0, 0, 0], np.uint8)
    else:
        color_weight = (to_color - from_color) / w
    for x in range(w):
        img[0:h, x] = add_color(from_color, color_weight * (x+1))

def gradation_height(img, from_color, to_color):
    h, w = img.shape[:2]
    if h <= 0:
        color_weight = np.array([0, 0, 0], np.uint8)
    else:
        color_weight = (to_color - from_color) / h
    for y in range(h):
        img[y, 0:w] = add_color(from_color, color_weight * (y+1))

def get_cosine_kernel(x_size, y_size=None):
    if y_size is None:
        return cosine(np.arange(x_size), fs=x_size-1, f=1)
    else:
        kernel = np.empty((x_size, y_size))
        for i in range(x_size):
            # x = min(i, x_fs-i-1)
            x = min(i, x_size - 1 - i)
            # x -= i
            for j in range(y_size):
                y = min(j, y_size - j - 1)
                val = min(x, y)
                if x < y:
                    # val -= x
                    val = (x_size-1) / 2 - x
                    # val = x
                kernel[i][j] = val
        return cosine(kernel, fs=x_size-1, f=1)

def cosine(x, fs, f=1):
    """
    1 ~ 0 사이의 Cosine wave
    :param x: x value
    :param fs: 진폭
    :param f: 주기
    :return:
    """
    # print(x)
    return (np.cos(2 * np.pi * f * (x / fs)) + 1) / 2

def average_color(img):
    b, g, r = cv2.split(img)

    avg_b = b.mean()
    avg_g = g.mean()
    avg_r = r.mean()

    return (avg_b, avg_g, avg_r)

def add_color(img1, img2):
    """
    OpenCV image color를 더한다. 255 이상의 값은 255, 0 이하의 값은 0으로 고정한다.(연산 결과를 0 ~ 255 사이의 값으로 유지)
    :param img1: 더하고자 하는 이미지
    :param img2: 더하고자 하는 이미지
    :return: 결과 이미지
    """
    added = img1 + img2
    return np.clip(added, 0, 255)


class TooManyFaces(Exception):
    pass

class NoFaces(Exception):
    pass


class Extractor:
    """
    이미지에서 각 요소를 추출할 수 있는 함수를 제공하는 인터페이스.
    """

    def extract_erythema(self, img,
                     tv_1=client.TrackBarValue.NORMAL,
                     tv_2=client.TrackBarValue.NORMAL,
                     tv_3=client.TrackBarValue.NORMAL,
                     tv_4=client.TrackBarValue.NORMAL):

        request_data = client.serialize_req_param_data(client.SkinFeature.ERYTHEMA, img, tv_1, tv_2, tv_3, tv_4, False)
        return client.request(request_data)

    def extract_pore(self, img,
                     tv_1=client.TrackBarValue.NORMAL,
                     tv_2=client.TrackBarValue.NORMAL,
                     tv_3=client.TrackBarValue.NORMAL,
                     tv_4=client.TrackBarValue.NORMAL):

        request_data = client.serialize_req_param_data(client.SkinFeature.PORE, img, tv_1, tv_2, tv_3, tv_4, False )
        return client.request(request_data)

    def extract_pigmentation(self, img,
                     tv_1=client.TrackBarValue.NORMAL,
                     tv_2=client.TrackBarValue.NORMAL,
                     tv_3=client.TrackBarValue.NORMAL,
                     tv_4=client.TrackBarValue.NORMAL):

        request_data = client.serialize_req_param_data(client.SkinFeature.PIGMENTATION, img, tv_1, tv_2, tv_3, tv_4, False)
        return client.request(request_data)

    def extract_wrinkle(self, img,
                     tv_1=client.TrackBarValue.NORMAL,
                     tv_2=client.TrackBarValue.NORMAL,
                     tv_3=client.TrackBarValue.NORMAL,
                     tv_4=client.TrackBarValue.NORMAL):

        request_data = client.serialize_req_param_data(client.SkinFeature.WRINKLE, img, tv_1, tv_2, tv_3, tv_4, False)
        return client.request(request_data)

    def extract_all(self, erythema_img, pore_img, pigmentation_img, wrinkle_img,
                     tv_1=client.TrackBarValue.NORMAL,
                     tv_2=client.TrackBarValue.NORMAL,
                     tv_3=client.TrackBarValue.NORMAL,
                     tv_4=client.TrackBarValue.NORMAL):
        self.extract_erythema(erythema_img, tv_1, tv_2, tv_3, tv_4)
        self.extract_pore(pore_img, tv_1, tv_2, tv_3, tv_4)
        self.extract_pigmentation(pigmentation_img, tv_1, tv_2, tv_3, tv_4)
        self.extract_wrinkle(wrinkle_img, tv_1, tv_2, tv_3, tv_4)

    def getFeatureData(self):
        data = dict()
        data['erythema'] = self.erythema
        data['pore'] = self.pore
        data['pigmentation'] = self.pigmentation
        data['wrinkle'] = self.wrinkle
        return data


class Analyzer:
    """
    추출된 요소의 갯수 등을 이용하여 점수화하는 함수를 제공한다.
    """
    min_score = {}
    max_score = {}

    def __init__(self):
        self.extractor = Extractor()

    def analyze_emotion(self, emotion_data, result_data=None):
        emotion_data['score_emotion'] = (emotion_data['score_emotion'] + 4) * 10
        if result_data is not None:
            emotion_data.update(result_data)
        return emotion_data

    def analyze_erythema(self, erythema_img, result_data=None):
        mapped_key = {'Count': 'erythema_num', 'Area': 'erythema_average_area', 'Darkness': 'erythema_darkness'}
        erythema_data = self.extractor.extract_erythema(erythema_img)
        erythema_data = change_key(erythema_data, mapped_key)
        num_erythema = erythema_data['erythema_num']
        avg_area = erythema_data['erythema_average_area']
        avg_darkness = erythema_data['erythema_darkness']
        erythema_data['score_erythema'] = get_random_normal()[0]
        #print('Erythema[ %d, %d, %d ]' % (num_erythema, avg_area, avg_darkness))
        if result_data is not None:
            erythema_data.update(result_data)
        return erythema_data

    def analyze_pore(self, pore_img, result_data=None):
        mapped_key = {'Count': 'pore_num'}
        pore_data = self.extractor.extract_pore(pore_img)
        pore_data = change_key(pore_data, mapped_key)
        num_pore = pore_data['pore_num']
        pore_data['score_pore'] = get_random_normal()[0]
        #print('Pore[ %d ]' % (num_pore))
        if result_data is not None:
            pore_data.update(result_data)
        return pore_data

    def analyze_pigmentation(self, pigmentation_img, result_data=None):
        mapped_key = {'Count': 'pigmentation_num', 'Area': 'pigmentation_average_area', 'Darkness': 'pigmentation_darkness'}
        pigmentation_data = self.extractor.extract_pigmentation(pigmentation_img)
        pigmentation_data = change_key(pigmentation_data, mapped_key)
        num_pigmentation = pigmentation_data['pigmentation_num']
        avg_area = pigmentation_data['pigmentation_average_area']
        avg_darkness = pigmentation_data['pigmentation_darkness']
        pigmentation_data['score_pigmentation'] = get_random_normal()[0]
        #print('Pigmentation[ %d, %d, %d ]' % (num_pigmentation, avg_area, avg_darkness))
        if result_data is not None:
            pigmentation_data.update(result_data)
        return pigmentation_data

    def analyze_wrinkle(self, wrinkle_img, result_data=None):
        mapped_key = {'Count': 'wrinkle_num', 'Area': 'wrinkle_average_area', 'Darkness': 'wrinkle_darkness',
                      'Pitch': 'wrinkle_pitch', 'Length': 'wrinkle_length'}
        wrinkle_data = self.extractor.extract_wrinkle(wrinkle_img)
        wrinkle_data = change_key(wrinkle_data, mapped_key)
        num_wrinkle = wrinkle_data['wrinkle_num']
        avg_area = wrinkle_data['wrinkle_average_area']
        avg_darkness = wrinkle_data['wrinkle_darkness']
        pitch = wrinkle_data['wrinkle_pitch']
        length = wrinkle_data['wrinkle_length']
        wrinkle_data['score_wrinkle'] = get_random_normal()[0]
        #print('Wrinkle[ %d, %d, %d, %d, %d ]' % (num_wrinkle, avg_area, avg_darkness, pitch, length))
        if result_data is not None:
            wrinkle_data.update(result_data)
        return wrinkle_data

    def calc_total_score(self, result_data):
        score_emotion = result_data['score_emotion']
        score_erythema = result_data['score_erythema']
        score_pore = result_data['score_pore']
        score_pigmentation = result_data['score_pigmentation']
        score_wrinkle = result_data['score_wrinkle']
        total_score = get_score_data(emotion=score_emotion, erythema=score_erythema,
                       pore=score_pore, pigmentation=score_pigmentation, wrinkle=score_wrinkle)
        result_data['score_total'] = total_score['score_total']
        return result_data



class CascadeDetector:
    """
    각 요소 추출을 위해 ROI를 찾는 기능을 제공한다.
    """
    def __get_center(self, left_top_pixel, right_bottom_pixel):
        ltx, lty = left_top_pixel
        rbx, rby = right_bottom_pixel
        return (int((ltx+rbx)/2), int((lty+rby)/2))


    def __get_best_feature_roi(self, roi_gray, feature_cascade, k_mean, k_mean_step=1, rate_of_change=0.5, scale=1.3):
        # Direction True: increment k_mean, False: Decrement k_mean
        prev_direction = True

        features = feature_cascade.detectMultiScale(roi_gray, scale, k_mean)
        while not len(features) == 1:
            curr_direction = len(features) > 1
            if prev_direction == curr_direction:
                # 진행방향(k_mean 증감)이 그대로인 경우(계속 증가 혹은 계속 감소): k_mean을 이전과 같은 k_mean_step으로 증가 혹은 감소시킴
                if k_mean_step < 1:
                    # k_mean 변화량이 0인 경우: 무한루프
                    if curr_direction:
                        k_mean += 1
                    else:
                        k_mean -= 1
                    if k_mean < 1:
                        k_mean = 1
                    features = feature_cascade.detectMultiScale(roi_gray, scale, k_mean)
                    if len(features) > 0:
                        features = [features[0]]
                    break
                k_mean += k_mean_step
                prev_direction = curr_direction

            else:
                # 진행방향이 바뀐 경우: k_mean 증감율을 -1 * (k_mean_step * rate_of_change)
                k_mean_step = math.ceil(k_mean_step * rate_of_change * -1)
                k_mean += k_mean_step
                prev_direction = curr_direction

            features = feature_cascade.detectMultiScale(roi_gray, scale, k_mean)

        return features

    def __filter_skin(self, img):
        # define the upper and lower boundaries of the HSV pixel
        # intensities to be considered 'skin'
        lower = np.array([0, 48, 80], dtype="uint8")
        upper = np.array([20, 255, 255], dtype="uint8")

        converted = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        skinMask = cv2.inRange(converted, lower, upper)

        # apply a series of erosions and dilations to the mask
        # using an elliptical kernel
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
        skinMask = cv2.erode(skinMask, kernel, iterations=2)
        skinMask = cv2.dilate(skinMask, kernel, iterations=2)

        # blur the mask to help remove noise, then apply the
        # mask to the frame
        skinMask = cv2.GaussianBlur(skinMask, (3, 3), 0)
        skin = cv2.bitwise_and(img, img, mask=skinMask)

        return skin

    def detect_face(self, img, use_gray=False, visible=False):
        face_cascade = cv2.CascadeClassifier(os.path.join(os.getcwd(), 'api', 'services', 'cascades', 'frontalFace.xml'))

        overlay = img.copy()
        gray = cv2.cvtColor(overlay, cv2.COLOR_BGR2GRAY)
        # cv2.imshow('Gray', gray)
        # cv2.waitKey()
        # cv2.destroyAllWindows()

        faces = self.__get_best_feature_roi(gray, face_cascade, k_mean=5, scale=1.3)
        for (x, y, w, h) in faces:
            face = img[y:y + h, x:x + w]

            cv2.rectangle(overlay, (x, y), (x + w, y + h), (255, 0, 0), 2)
            face_center = self.__get_center(left_top_pixel=(x, y), right_bottom_pixel=(x + w, y + h))
            cv2.circle(overlay, face_center, 2, (255, 0, 0), -1)

            if visible:
                h, w = overlay.shape[:2]
                if h > 1200 or w > 1200:
                    img = cv2.resize(overlay, dsize=None, fx=0.4, fy=0.4, interpolation=cv2.INTER_AREA)
                cv2.imshow('Facial Features', overlay)
                # cv2.imshow('Mask', self.__filter_skin(face_img))
                cv2.waitKey()
                cv2.destroyAllWindows()
        if faces is None or len(faces) == 0:
            face = gray
        if use_gray and face.shape[2] == 3:
            face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)

        return face


    def detect_facial_feature(self, img, visible=False):
        cascades_dir = os.path.join(os.getcwd(), 'api', 'services', 'cascades')
        face_cascade = cv2.CascadeClassifier(os.path.join(cascades_dir, 'frontalFace.xml'))
        eye_cascade = cv2.CascadeClassifier(os.path.join(cascades_dir, 'Eyes_cascade.xml'))
        nose_cascade = cv2.CascadeClassifier(os.path.join(cascades_dir, 'Nose_cascade.xml'))
        mouth_cascade = cv2.CascadeClassifier(os.path.join(cascades_dir, 'Mouth_cascade.xml'))

        roi_data = {}

        overlay = img.copy()
        gray = cv2.cvtColor(overlay, cv2.COLOR_BGR2GRAY)

        faces = self.__get_best_feature_roi(gray, face_cascade, k_mean=5, scale=1.3)
        for (x, y, w, h) in faces:
            face = img[y:y + h, x:x + w]
            roi_data['face'] = face
            roi_gray = gray[y:y + h, x:x + w]
            roi_color = overlay[y:y + h, x:x + w]
            face_img = roi_color.copy()

            cv2.rectangle(overlay, (x, y), (x+w, y+h), (255, 0, 0), 2)
            face_center = self.__get_center(left_top_pixel=(x, y), right_bottom_pixel=(x+w, y+h))
            cv2.circle(overlay, face_center, 2, (255, 0, 0), -1)
            print('Face: ', face_center)

            eyes = self.__get_best_feature_roi(roi_gray, eye_cascade, k_mean=3, scale=1.3)
            for (ex, ey, ew, eh) in eyes:
                roi_data['eyes'] = face[ey:ey + eh, ex:ex + ew]
                cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2)
                #cv2.circle(roi_color, (int((ey+eh)/2), int((ex+ew)/2)), 2, (0, 255, 0), -1)
                eyes_center = self.__get_center(left_top_pixel=(ex, ey), right_bottom_pixel=(ex+ew, ey+eh))
                cv2.circle(roi_color, eyes_center, 2, (0, 255, 0), -1)
                print('Eyes: ', eyes_center)

            nose = self.__get_best_feature_roi(roi_gray, nose_cascade, k_mean=3, scale=1.3)
            for (nx, ny, nw, nh) in nose:
                roi_data['nose'] = face[ny:ny + nh, nx:nx + nw]
                cv2.rectangle(roi_color, (nx, ny), (nx+nw, ny+nh), (0, 0, 255), 2)
                nose_center = self.__get_center(left_top_pixel=(nx, ny), right_bottom_pixel=(nx+nw, ny+nh))
                cv2.circle(roi_color, nose_center, 2, (0, 0, 255), -1)
                print('Nose: ', nose_center)

            mouth = self.__get_best_feature_roi(roi_gray, mouth_cascade, k_mean=50, k_mean_step=10, scale=1.3)
            for (mx, my, mw, mh) in mouth:
                roi_data['mouth'] = face[my:my + mh, mx:mx + mw]
                cv2.rectangle(roi_color, (mx, my), (mx+mw, my+mh), (255, 255, 0), 2)
                mouth_center = self.__get_center(left_top_pixel=(mx, my), right_bottom_pixel=(mx+mw, my+mh))
                cv2.circle(roi_color, mouth_center, 2, (255, 255, 0), -1)
                print('Mouth: ', mouth_center)

            if len(eyes) == 1 and len(nose):
                ex, ey, ew, eh = eyes[0]
                nx, ny, nw, nh = nose[0]
                roi_data['right_cheek'] = face[ey+eh:int((ny+ny+nh) / 2), ex:nx]
                roi_data['left_cheek'] = face[ey+eh:int((ny+ny+nh) / 2), nx+nw:ex+ew]
                cv2.rectangle(roi_color, (ex, ey+eh), (nx, int((ny+ny+nh) / 2)), (255, 0, 255), 2)
                cv2.rectangle(roi_color, (nx+nw, ey+eh), (ex+ew, int((ny+ny+nh) / 2)), (255, 0, 255), 2)

            if visible:
                h, w = overlay.shape[:2]
                if h > 1200 or w > 1200:
                    img = cv2.resize(overlay, dsize=None, fx=0.4, fy=0.4, interpolation=cv2.INTER_AREA)
                cv2.imshow('Facial Features', overlay)
                #cv2.imshow('Mask', self.__filter_skin(face_img))
                cv2.waitKey()
                cv2.destroyAllWindows()

        return roi_data


class LandmarkDetector:
    """
        각 요소 추출을 위해 ROI를 찾는 기능을 제공한다.
    """
    def get_landmarks(self, img):
        PREDICTIOR_PATH = os.path.join(os.getcwd(), 'api', 'services', 'cascades',
                                       'shape_predictor_68_face_landmarks.dat')
        predictor = dlib.shape_predictor(PREDICTIOR_PATH)
        detector = dlib.get_frontal_face_detector()

        rects = detector(img, 1)

        if len(rects) > 1:
            raise TooManyFaces
        if len(rects) < 1:
            raise NoFaces

        return np.matrix([[p.x, p.y] for p in predictor(img, rects[0]).parts()])

    def matrix2dict(self, landmarks):
        featuers = {}
        featuers['jaw'] = landmarks[0:17]
        featuers['right_eyebrow'] = landmarks[17:22]
        featuers['left_eyebrow'] = landmarks[22:27]
        featuers['nose'] = landmarks[27:36]
        featuers['right_eye'] = landmarks[36:42]
        featuers['left_eye'] = landmarks[42:48]
        featuers['mouth'] = landmarks[48:68]
        return featuers

    def dict2matrix(self, landmarks):
        features = np.append(landmarks['jaw'], landmarks['right_eyebrow'], axis=0)
        features = np.append(features, landmarks['left_eyebrow'], axis=0)
        features = np.append(features, landmarks['nose'], axis=0)
        features = np.append(features, landmarks['right_eye'], axis=0)
        features = np.append(features, landmarks['left_eye'], axis=0)
        features = np.append(features, landmarks['mouth'], axis=0)

        return features

    def annotate_landmarks(self, img, landmarks):
        overlay = img.copy()
        for idx, point in enumerate(landmarks):
            pos = (point[0, 0], point[0, 1])
            cv2.putText(overlay, str(idx), pos, fontFace=cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, fontScale=0.4, color=(0, 0, 255))
            cv2.circle(overlay, pos, 3, color=(0, 255, 255))

        return overlay

    def calc_appended_image_size(self, img_points):
        # print(img_points)
        width = np.max(img_points, axis=0)[2]
        height = 0
        for i, pts in enumerate(img_points):
            height += pts[3]

        return (height, width)


    def expand_eye(self, eyes, x_rate=1.6, y_rate=2.0):
        """
        눈 영역을 확장한다.
        :param eyes: eye landmark(points array)
        :param x_rate: x축 확장 비율
        :param y_rate: y축 확장 비율
        :return: 확장된 eye landmark(points array)
        """
        result = eyes.copy()

        x_half_length = math.ceil((eyes[3][0, 0] - eyes[0][0, 0]) / 2)
        x_center = eyes[3][0, 0] - x_half_length
        x2_half_length = math.ceil((eyes[2][0, 0] - eyes[1][0, 0]) / 2)
        x2_center = eyes[2][0, 0] - x2_half_length
        x3_half_length = math.ceil((eyes[4][0, 0] - eyes[5][0, 0]) / 2)
        x3_center = eyes[4][0, 0] - x3_half_length
        y1_half_length = math.ceil((eyes[5][0, 1] - eyes[1][0, 1]) / 2)
        y1_center = eyes[1][0, 1] + y1_half_length
        y2_half_length = math.ceil((eyes[4][0, 1] - eyes[2][0, 1]) / 2)
        y2_center = eyes[2][0, 1] + y2_half_length

        result[0][0, 0] = x_center - x_half_length * x_rate
        result[1][0, 0] = x2_center - x2_half_length * x_rate
        result[1][0, 1] = y1_center - y1_half_length * y_rate
        result[2][0, 0] = x2_center + x2_half_length * x_rate
        result[2][0, 1] = y2_center - y2_half_length * y_rate
        result[3][0, 0] = x_center + x_half_length * x_rate
        result[4][0, 0] = x3_center + x3_half_length * x_rate
        result[4][0, 1] = y2_center + y2_half_length * y_rate
        result[5][0, 0] = x3_center - x3_half_length * x_rate
        result[5][0, 1] = y1_center + y1_half_length * y_rate

        return result

    def expand_eye_brow(self, eye_brows, x_rate=1.1, y_rate=1.3):
        """
        눈썹 영역을 확장한다.
        :param eye_brows: eye_brow landmark(points array)
        :param x_rate: x축 확장 비율
        :param y_rate: y축 확장 비율
        :return: 확장된 eye_brow landmark(points array)
        """
        result = eye_brows.copy()

        x_half_length = math.ceil((eye_brows[4][0, 0] - eye_brows[0][0, 0]) / 2)
        x_center = eye_brows[4][0, 0] - x_half_length
        x2_half_length = math.ceil((eye_brows[3][0, 0] - eye_brows[1][0, 0]) / 2)
        x2_center = eye_brows[3][0, 0] - x2_half_length
        y_low = eye_brows[0][0, 1] if eye_brows[0][0, 1] > eye_brows[4][0, 1] else eye_brows[4][0, 1]
        y1_length = y_low - eye_brows[1][0, 1]
        y2_length = y_low - eye_brows[2][0, 1]
        y3_length = y_low - eye_brows[3][0, 1]

        result[0][0, 0] = x_center - x_half_length * x_rate
        result[1][0, 0] = x2_center - x2_half_length * x_rate
        result[1][0, 1] = y_low - y1_length * y_rate
        result[2][0, 1] = y_low - y2_length * y_rate
        result[3][0, 0] = x2_center + x2_half_length * x_rate
        result[3][0, 1] = y_low - y3_length * y_rate
        result[4][0, 0] = x_center + x_half_length * x_rate

        return result

    def expand_nose(self, nose, x_rate=1.3, y_rate=1.0):
        """
        코 영역을 확장한다.
        :param nose: nose landmark(points array)
        :param x_rate: x축 확장 비율
        :param y_rate: y축 확장 비율
        :return: 확장된 nose landmark(points array)
        """
        result = nose.copy()

        x1_half_length = math.ceil((nose[8][0, 0] - nose[4][0, 0]) / 2)
        x1_center = nose[8][0, 0] - x1_half_length
        x2_half_length = math.ceil((nose[7][0, 0] - nose[5][0, 0]) / 2)
        x2_center = nose[7][0, 0] - x2_half_length

        y_top = nose[0][0, 1]
        y1_length = nose[4][0, 1] - y_top
        y2_length = nose[5][0, 1] - y_top
        y3_length = nose[6][0, 1] - y_top
        y4_length = nose[7][0, 1] - y_top
        y5_length = nose[8][0, 1] - y_top

        result[4][0, 0] = x1_center - x1_half_length * x_rate
        result[4][0, 1] = y_top + y1_length * y_rate
        result[5][0, 0] = x2_center - x2_half_length * x_rate
        result[5][0, 1] = y_top + y2_length * y_rate
        result[6][0, 1] = y_top + y3_length * y_rate
        result[7][0, 0] = x2_center + x2_half_length * x_rate
        result[7][0, 1] = y_top + y4_length * y_rate
        result[8][0, 0] = x1_center + x1_half_length * x_rate
        result[8][0, 1] = y_top + y5_length * y_rate

        return result

    def expand_mouth(self, mouth, x_rate=1.2, y_rate=1.2):
        """
        입 영역을 확장한다.
        :param mouth: mouth landmark(points array)
        :param x_rate: x축 확장 비율
        :param y_rate: y축 확장 비율
        :return: 확장된 mouth landmark(points array)
        """
        result = mouth.copy()

        x1_half_length = math.ceil((mouth[6][0, 0] - mouth[0][0, 0]) / 2)
        x1_center = mouth[6][0, 0] - x1_half_length
        x2_half_length = math.ceil((mouth[5][0, 0] - mouth[1][0, 0]) / 2)
        x2_center = mouth[5][0, 0] - x2_half_length
        x3_half_length = math.ceil((mouth[4][0, 0] - mouth[2][0, 0]) / 2)
        x3_center = mouth[4][0, 0] - x3_half_length
        x4_half_length = math.ceil((mouth[7][0, 0] - mouth[11][0, 0]) / 2)
        x4_center = mouth[7][0, 0] - x4_half_length
        x5_half_length = math.ceil((mouth[8][0, 0] - mouth[10][0, 0]) / 2)
        x5_center = mouth[8][0, 0] - x5_half_length
        y1_half_length = math.ceil((mouth[11][0, 1] - mouth[1][0, 1]) / 2)
        y1_center = mouth[1][0, 1] + y1_half_length
        y2_half_length = math.ceil((mouth[10][0, 1] - mouth[2][0, 1]) / 2)
        y2_center = mouth[2][0, 1] + y2_half_length
        y3_half_length = math.ceil((mouth[9][0, 1] - mouth[3][0, 1]) / 2)
        y3_center = mouth[3][0, 1] + y3_half_length
        y4_half_length = math.ceil((mouth[8][0, 1] - mouth[4][0, 1]) / 2)
        y4_center = mouth[4][0, 1] + y4_half_length
        y5_half_length = math.ceil((mouth[7][0, 1] - mouth[5][0, 1]) / 2)
        y5_center = mouth[5][0, 1] + y5_half_length

        result[0][0, 0] = x1_center - x1_half_length * x_rate
        result[1][0, 0] = x2_center - x2_half_length * x_rate
        result[1][0, 1] = y1_center - y1_half_length * y_rate
        result[2][0, 0] = x3_center - x3_half_length * x_rate
        result[2][0, 1] = y2_center - y2_half_length * y_rate
        result[3][0, 1] = y3_center - y3_half_length * y_rate
        result[4][0, 0] = x3_center + x3_half_length * x_rate
        result[4][0, 1] = y4_center - y4_half_length * y_rate
        result[5][0, 0] = x2_center + x2_half_length * x_rate
        result[5][0, 1] = y5_center - y5_half_length * y_rate
        result[6][0, 0] = x1_center + x1_half_length * x_rate
        result[7][0, 0] = x4_center + x4_half_length * x_rate
        result[7][0, 1] = y5_center + y5_half_length * y_rate
        result[8][0, 0] = x5_center + x5_half_length * x_rate
        result[8][0, 1] = y4_center + y4_half_length * y_rate
        result[9][0, 1] = y3_center + y3_half_length * y_rate
        result[10][0, 0] = x5_center - x5_half_length * x_rate
        result[10][0, 1] = y2_center + y2_half_length * y_rate
        result[11][0, 0] = x4_center - x4_half_length * x_rate
        result[11][0, 1] = y1_center + y1_half_length * y_rate

        return result

    def draw_landmarks(self, im, landmarks, colors=None, alpha=0.75,
                       use_feature=['jaw', 'right_eyebrow', 'left_eyebrow', 'nose', 'right_eye', 'left_eye', 'mouth']):
        # create two copies of the input image -- one for the
        # overlay and one for the final output image
        overlay = im.copy()

        # if the colors list is None, initialize it with a unique
        # color for each facial landmark region
        if colors is None:
            colors = [(19, 199, 109), (79, 76, 240), (230, 159, 23),
                      (168, 100, 168), (158, 163, 32),
                      (163, 38, 32), (180, 42, 220)]

        facial_landmarks = self.matrix2dict(landmarks)
        for i, feature in enumerate(use_feature):
            pts = facial_landmarks[feature]
            if feature == 'jaw':
                for idx in range(1, len(pts)):
                    ptA = (pts[idx - 1][0, 0], pts[idx - 1][0, 1])
                    ptB = (pts[idx][0, 0], pts[idx][0, 1])
                    cv2.line(overlay, ptA, ptB, colors[i], 2)
                    cv2.circle(overlay, ptB, 3, color=(0, 255, 255))

            else:
                hull = cv2.convexHull(pts)
                cv2.drawContours(overlay, [hull], -1, colors[i], -1)

                for idx in range(len(pts)):
                    # ptA = tuple(pts[idx - 1])
                    # ptB = tuple(pts[idx])
                    pos = (pts[idx][0, 0], pts[idx][0, 1])
                    cv2.circle(overlay, pos, 3, color=(0, 255, 255))

        cv2.imshow('Overlay_Landmarks', overlay)
        cv2.waitKey()
        cv2.destroyAllWindows()

    # def expand_landmarks_from_img(self, img, landmarks, color=(0,0,0),
    #                   use_feature=['jaw', 'right_eyebrow', 'left_eyebrow', 'nose', 'right_eye', 'left_eye', 'mouth']):
    #     # create two copies of the input image -- one for the
    #     # overlay and one for the final output image
    #     overlay = img.copy()
    #
    #     facial_landmarks = self.matrix2dict(landmarks)
    #     for i, feature in enumerate(use_feature):
    #         pts = facial_landmarks[feature]
    #         if feature == 'jaw':
    #             for idx in range(1, len(pts)):
    #                 ptA = (pts[idx - 1][0, 0], pts[idx - 1][0, 1])
    #                 ptB = (pts[idx][0, 0], pts[idx][0, 1])
    #                 cv2.line(overlay, ptA, ptB, color, 2)
    #                 cv2.circle(overlay, ptB, 3, color=(0, 255, 255))
    #
    #         else:
    #             if feature == 'right_eye' or feature == 'left_eye':
    #                 pts = self.expand_eye(facial_landmarks[feature])
    #             elif feature == 'right_eyebrow' or feature == 'left_eyebrow':
    #                 pts = self.expand_eye_brow(facial_landmarks[feature])
    #             elif feature == 'mouth':
    #                 pts = self.expand_mouth(facial_landmarks[feature])
    #             elif feature == 'nose':
    #                 pts = self.expand_nose(facial_landmarks[feature], x_rate=1.5)
    #             hull = cv2.convexHull(pts)
    #             cv2.drawContours(overlay, [hull], -1, color, -1)
    #
    #             for idx in range(len(pts)):
    #                 pos = (pts[idx][0, 0], pts[idx][0, 1])
    #                 cv2.circle(overlay, pos, 3, color=(0, 255, 255))
    #
    #     landmark_area_min_x, landmark_area_min_y, landmark_area_max_x, landmark_area_max_y = self.range_from_landmark(landmarks)
    #     overlay = overlay[landmark_area_min_y:landmark_area_max_y, landmark_area_min_x:landmark_area_max_x]
    #
    #     cv2.imshow('Remove_Landmarks', overlay)
    #     cv2.waitKey()
    #     cv2.destroyAllWindows()
    #
    #     return overlay

    def remove_feature(self, face, feature_points, fill_color, fill_area_rate=0.3, mean_range=10):
        (fx, fy, fw, fh) = feature_points
        feature = face[fy:fy + fh, fx:fx + fw]

        # feature 영역에서 fill_area_rate 만큼의 영역을 fill_color로 채움. 채우는 영역은 feature 영역의 중심
        min_rate = (1-fill_area_rate) / 2
        max_rate = min_rate + fill_area_rate
        fill_area_min_x = int(fw * min_rate)
        fill_area_min_y = int(fh * min_rate)
        fill_area_max_x = int(fw * max_rate)
        fill_area_max_y = int(fh * max_rate)
        fill_area = np.zeros((fill_area_max_y-fill_area_min_y, fill_area_max_x-fill_area_min_x, 3), np.uint8)
        fill_area[:] = fill_color
        feature[fill_area_min_y:fill_area_max_y, fill_area_min_x:fill_area_max_x] = fill_area

        # X axis
        prev_x_mean = np.mean(face[fy:fy + fh, fx - mean_range:fx - 1], axis=1)
        next_x_mean = np.mean(face[fy:fy + fh, fx + fw + 1:fx + fw + mean_range], axis=1)

        # Y axis
        prev_y_mean = np.mean(face[fy - mean_range:fy - 1, fx:fx + fw], axis=0)
        next_y_mean = np.mean(face[fy + fh + 1:fy + fh + mean_range, fx:fx + fw], axis=0)

        # for i in range(fill_area_min_x):
        gradation_width(face[fy+fill_area_min_y:fy+fill_area_max_y, fx:fx+fill_area_min_x],
                        from_color=prev_x_mean[fill_area_min_y:fill_area_max_y],
                        to_color=face[fy+fill_area_min_y:fy+fill_area_max_y, fx+fill_area_min_x])
        # for i in range(fill_area_max_x+1, fx+fw):
        gradation_width(face[fy+fill_area_min_y:fy+fill_area_max_y, fx+fill_area_max_x:fx+fw],
                        from_color=face[fy+fill_area_min_y:fy+fill_area_max_y, fx+fill_area_max_x-1],
                        to_color=next_x_mean[fill_area_min_y:fill_area_max_y])

        # for j in range(fill_area_min_y):
        gradation_height(face[fy:fy+fill_area_min_y, fx+fill_area_min_x:fx+fill_area_max_x],
                         from_color=prev_y_mean[fill_area_min_x:fill_area_max_x],
                         to_color=face[fy+fill_area_min_y, fx+fill_area_min_x:fx+fill_area_max_x])
        # for j in range(fill_area_max_y+1, fy+fh):
        gradation_height(face[fy+fill_area_max_y:fy+fh, fx + fill_area_min_x:fx + fill_area_max_x],
                         from_color=face[fy+fill_area_max_y-1, fx+fill_area_min_x:fx+fill_area_max_x],
                         to_color=next_y_mean[fill_area_min_x:fill_area_max_x])

        # print(feature_points)
        # print(prev_x_mean.shape, prev_y_mean.shape)
        # print(next_x_mean.shape, next_y_mean.shape)

        for i in range(fw):
            if i < fill_area_min_x:
                # Top-Left side
                mean_y_upper = fill_area_min_y - (i+1)
                mean_x_upper = fill_area_min_x - (i+1)
                # if mean_y_upper < 1:
                #     mean_y_upper = 1
                mean_value_upper = np.mean(np.array([face[fy+mean_y_upper + 1][fx+mean_x_upper],
                                                     face[fy+mean_y_upper][fx+mean_x_upper + 1]]), axis=0)
                #face[fy+mean_y_upper][fx+mean_x_upper] = mean_value_upper
                gradation_height(np.reshape(face[fy:fy+mean_y_upper+1, fx+mean_x_upper], (mean_y_upper+1, 1, 3)),
                                 from_color=prev_y_mean[mean_x_upper],
                                 to_color=mean_value_upper)
                if mean_y_upper >= 0:
                    gradation_width(np.reshape(face[fy+mean_y_upper, fx:fx+mean_x_upper+1], (1, mean_x_upper+1, 3)),
                                    from_color=prev_x_mean[mean_y_upper],
                                    to_color=mean_value_upper)
                # print('(%d, %d)' % (fy+mean_y_upper+1, fx+mean_x_upper),face[fy+mean_y_upper+1][fx+mean_x_upper],
                #       '(%d, %d)' % (fy+mean_y_upper, fx+mean_x_upper+1), face[fy+mean_y_upper][fx+mean_x_upper+1])
                # print('Mean X: (%d, %d)' % (fy+mean_y_upper, fx+mean_x_upper), face[fy+mean_y_upper][fx+mean_x_upper])

                # Bottom-Left side
                mean_y_lower = fill_area_max_y + i
                mean_x_lower = mean_x_upper
                mean_value_lower =  np.mean(np.array([face[fy+mean_y_lower-1][fx+mean_x_lower],
                                                     face[fy+mean_y_lower][fx+mean_x_lower+1]]), axis=0)
                gradation_height(np.reshape(face[fy+mean_y_lower-1:fy+fh, fx + mean_x_lower], ((fh-mean_y_lower)+1, 1, 3)),
                                 from_color=mean_value_lower,
                                 to_color=next_y_mean[mean_x_lower])
                if mean_y_lower < prev_x_mean.shape[0]:
                    gradation_width(np.reshape(face[fy+mean_y_lower, fx:fx+mean_x_lower+1], (1, mean_x_lower+1, 3)),
                                    from_color=prev_x_mean[mean_y_lower],
                                    to_color=mean_value_lower)
                # print('(%d, %d)' % (fy+mean_y_lower-1, fx+mean_x_lower), face[fy + mean_y_lower-1][fx + mean_x_lower],
                #       '(%d, %d)' % (fy+mean_y_lower, fx+mean_x_lower+1), face[fy + mean_y_lower][fx + mean_x_lower+1])
                # print('Mean Y: (%d, %d)'%(fy+mean_y_lower, fx+mean_x_lower), face[fy+mean_y_lower][fx+mean_x_lower])

            elif i >= fill_area_max_x:
                # Top-Right side
                mean_y_upper = fill_area_min_y - ((i-fill_area_max_x) + 1)
                mean_x_upper = i
                mean_value_upper = np.mean(np.array([face[fy + mean_y_upper+1][fx + mean_x_upper],
                                                     face[fy + mean_y_upper][fx + mean_x_upper-1]]), axis=0)
                gradation_height(np.reshape(face[fy:fy+mean_y_upper+1, fx+mean_x_upper], (mean_y_upper+1, 1, 3)),
                                 from_color=prev_y_mean[mean_x_upper],
                                 to_color=mean_value_upper)
                if mean_y_upper >= 0:
                    gradation_width(np.reshape(face[fy+mean_y_upper, fx+mean_x_upper-1:fx+fw], (1, (fw-mean_x_upper)+1, 3)),
                                    from_color=mean_value_upper,
                                    to_color=next_x_mean[mean_y_upper])

                # Bottom-Right side
                mean_y_lower = fill_area_max_y + (i-fill_area_max_x)
                mean_x_lower = mean_x_upper
                mean_value_lower = np.mean(np.array([face[fy+mean_y_lower-1][fx+mean_x_lower],
                                                     face[fy+mean_y_lower][fx+mean_x_lower-1]]), axis=0)
                gradation_height(np.reshape(face[fy+mean_y_lower-1:fy+fh, fx+mean_x_lower], ((fh-mean_y_lower)+1, 1, 3)),
                                 from_color=mean_value_lower,
                                 to_color=next_y_mean[mean_x_lower])
                if mean_y_lower < next_x_mean.shape[0]:
                    gradation_width(np.reshape(face[fy+mean_y_lower, fx+mean_x_lower-1:fx+fw], (1, (fw-mean_x_lower)+1, 3)),
                                    from_color=mean_value_lower,
                                    to_color=next_x_mean[mean_y_lower])

        # cv2.imshow('Face', face)
        # cv2.waitKey()
        # cv2.destroyAllWindows()
        return face


    def remove_gradation_features(self, face, feature_points, mean_range=10):
        (fx, fy, fw, fh) = feature_points
        feature = face[fy:fy+fh, fx:fx+fw]
        feature_1 = feature.copy()
        feature_2 = feature.copy()

        # X axis
        prev_x_mean = np.mean(face[fy:fy+fh, fx - mean_range:fx - 1], axis=1)
        next_x_mean = np.mean(face[fy:fy+fh, fx+fw+1:fx+fw + mean_range], axis=1)
        prev_x = face[fy:fy+fh, fx - 1]
        next_x = face[fy:fy+fh, fx+fw + 1]
        diff_x = (next_x_mean - prev_x_mean) / fw
        diff_x_rev = (prev_x_mean - next_x_mean) / fw

        # Y axis
        prev_y_mean = np.mean(face[fy-mean_range:fy-1, fx:fx+fw], axis=0)
        next_y_mean = np.mean(face[fy+fh+1:fy+fh+mean_range, fx:fx+fw], axis=0)
        prev_y = face[fy-1, fx:fx+fw]
        next_y = face[fy+fh+1, fx:fx+fw]
        diff_y = (next_y_mean - prev_y_mean) / fh
        diff_y_rev = (prev_y_mean - next_y_mean) / fh

        # Color Gradation
        feature_x = feature_1.copy()
        for i in range(fw):
            feature_1[:, i] = add_color(prev_x, (diff_x * (i + 1)))
            feature_2[:, (fw - 1) - i] = add_color(next_x, (diff_x_rev * (i + 1)))
            feature_x = cv2.addWeighted(feature_1, float(1 - (i / fw)), feature_2, float(i / fw), 1)
        blur_x = cv2.blur(feature_x, (9, 9))
        face[fy:fy+fh, fx:fx+fw] = blur_x

        feature_y = feature_2.copy()
        for j in range(fh):
            feature_1[j, :] = add_color(prev_y, (diff_y * (j + 1)))
            feature_2[(fh - 1) - j, :] = add_color(next_y, (diff_y_rev * (j + 1)))
            feature_y = cv2.addWeighted(feature_1, float(1 - (j / fh)), feature_2, float(j / fh), 1)
        blur_y = cv2.blur(feature_y, (9, 9))
        face[fy:fy+fh, fx:fx+fw] = blur_y

        # Add X_Gradation and Y_Gradation
        weight = get_cosine_kernel(fw)
        for i in range(fw):
            pow_val = (weight[i] + 3) ** 2
            feature_1[:, i] = cv2.addWeighted(feature_x[:, i], float(np.power(weight[i], pow_val)),
                                              feature_y[:, i], float(1 - np.power(weight[i], pow_val)), 0)
        face[fy:fy+fh, fx:fx+fw] = feature_1
        blur = cv2.medianBlur(face[fy-mean_range:fy+fh+mean_range, fx-mean_range:fx+fw+mean_range], 15)
        face[fy - mean_range:fy + fh + mean_range, fx - mean_range:fx + fw + mean_range] = blur

        # cv2.imshow('Gradation', face)
        # cv2.waitKey()
        # cv2.destroyAllWindows()
        return face

    def remove_eyes(self, img, landmark, face_pts):
        x, y, w, h = face_pts[0]
        expanded_landmark = self.expand_eye(landmark, x_rate=1.5, y_rate=1.5)
        min_x, min_y, max_x, max_y = self.range_from_landmark(expanded_landmark)
        eye_pts = (min_x, min_y, max_x-min_x, max_y-min_y)
        # face = self.remove_features(img, eye_pts)
        face = self.remove_feature(img, eye_pts, np.array([200, 200, 200], np.uint8))
        return face[y:y+h, x:x+w]

    def remove_nose(self, img, landmark, face_pts):
        x, y, w, h = face_pts[0]
        expanded_landmark = self.expand_nose(landmark, x_rate=1.5, y_rate=1.0)
        min_x, min_y, max_x, max_y = self.range_from_landmark(expanded_landmark)
        nose_pts = (min_x, min_y, max_x - min_x, max_y - min_y)
        # face = self.remove_features(img, nose_pts)
        face = self.remove_feature(img, nose_pts, np.array([200, 200, 200], np.uint8))
        return face[y:y + h, x:x + w]

    def remove_mouth(self, img, landmark, face_pts):
        x, y, w, h = face_pts[0]
        expanded_landmark = self.expand_mouth(landmark, x_rate=1.1, y_rate=1.0)
        min_x, min_y, max_x, max_y = self.range_from_landmark(expanded_landmark)
        mouth_pts = (min_x, min_y, max_x - min_x, max_y - min_y)
        # face = self.remove_features(img, mouth_pts)
        face = self.remove_feature(img, mouth_pts, np.array([200, 200, 200], np.uint8))
        return face[y:y + h, x:x + w]


    def detect_facial_feature(self, img, overlay=None, visible=False, colors=None,
                      use_feature=['jaw', 'right_eyebrow', 'left_eyebrow', 'nose', 'right_eye', 'left_eye', 'mouth']):
        if overlay is None:
            overlay = img.copy()

        if colors is None:
            colors = [(19, 199, 109), (79, 76, 240), (230, 159, 23),
                      (168, 100, 168), (158, 163, 32),
                      (163, 38, 32), (180, 42, 220)]

        landmarks = self.matrix2dict(self.get_landmarks(img))

        features={}
        points = {}

        removed_landmarks_img = img.copy()
        skin = None
        pore_rois = None
        wrinkle_rois = None

        glabella_min_x, glabella_min_y, glabella_max_x, glabella_max_y, glabella_mid_y = (-1, -1, -1, -1, 0)
        right_cheek_min_x, right_cheek_min_y, right_cheek_max_x, right_cheek_max_y = (-1, -1, -1, -1)
        left_cheek_min_x, left_cheek_min_y, left_cheek_max_x, left_cheek_max_y = (-1, -1, -1, -1)
        face_min_x, face_min_y, face_max_x, face_max_y = (-1, -1, -1, -1)

        for i, feature in enumerate(use_feature):
            min_x, min_y, max_x, max_y = self.range_from_landmark(landmarks[feature])
            points[feature] = [(min_x, min_y, max_x-min_x, max_y-min_y)]
            features[feature] = img[min_y:max_y, min_x:max_x]
            cv2.rectangle(overlay, (min_x, min_y), (max_x, max_y), colors[i], 2)

            if feature == 'jaw':
                face_min_x, face_min_y, face_max_x, face_max_y = self.range_from_landmark(landmarks[feature])
            elif feature == 'right_eye':
                # 오른쪽 눈가 이미지 추가
                side_min_x = min_x - int((max_x - min_x) * 0.7)
                side_max_x = min_x
                points['right_eye_side'] = [(side_min_x, min_y, side_max_x - side_min_x, max_y - min_y)]
                features['right_eye_side'] = img[min_y:max_y, side_min_x:side_max_x]
                right_eye_side = np.array(points['right_eye_side'])
                if wrinkle_rois is not None:
                    wrinkle_rois = np.vstack((wrinkle_rois, right_eye_side))
                else:
                    wrinkle_rois = right_eye_side
                cv2.rectangle(overlay, (side_min_x, min_y), (side_max_x, max_y), (255, 255, 0), 2)
                # 오른쪽 볼 이미지 추가 사전 작업(1/2)
                pts = landmarks[feature]
                right_cheek_min_x = side_min_x #pts[0][0, 0]
                right_cheek_max_x = min(pts[2][0, 0], pts[4][0, 0])
            elif feature == 'left_eye':
                # 왼쪽 눈가 이미지 추가
                side_min_x = max_x
                side_max_x = max_x + int((max_x - min_x) * 0.7)
                points['left_eye_side'] = [(side_min_x, min_y, side_max_x - side_min_x, max_y - min_y)]
                features['left_eye_side'] = img[min_y:max_y, side_min_x:side_max_x]
                left_eye_side = np.array(points['left_eye_side'])
                if wrinkle_rois is not None:
                    wrinkle_rois = np.vstack((wrinkle_rois, left_eye_side))
                else:
                    wrinkle_rois = left_eye_side
                cv2.rectangle(overlay, (side_min_x, min_y), (side_max_x, max_y), (255, 255, 0), 2)
                # 왼쪽 볼 이미지 추가 사전 작업(1/2)
                pts = landmarks[feature]
                left_cheek_min_x = max(pts[1][0, 0], pts[5][0, 0])
                left_cheek_max_x = side_max_x  # pts[3][0, 0]
            elif feature == 'right_eyebrow':
                # 미간 이미지 추가 사전 작업(1/3)
                glabella_min_x = max_x
                glabella_mid_y = max(glabella_mid_y, max_y - int((max_y - min_y) / 2))
            elif feature == 'left_eyebrow':
                # 미간 이미지 추가 사전 작업(2/3)
                glabella_max_x = min_x
                glabella_mid_y = max(glabella_mid_y, max_y - int((max_y - min_y) / 2))
            elif feature == 'nose':
                # 미간 이미지 추가 사전 작업(3/3)
                glabella_min_y = glabella_mid_y - (min_y - glabella_mid_y)
                glabella_max_y = min_y
                # 오른쪽/왼쪽 볼 이미지 추가 사전 작업(2/2)
                pts = landmarks[feature]
                right_cheek_min_y = left_cheek_min_y = int((pts[1][0, 1] + pts[2][0, 1]) / 2)
                right_cheek_max_y = left_cheek_max_y = pts[6][0, 1]
                # 모공 추출용 코 이미지 추가
                nose_for_pore = pts[2:]
                min_nx, min_ny, max_nx, max_ny = self.range_from_landmark(nose_for_pore)
                points['nose_for_pore'] = [(min_nx, min_ny, max_nx - min_nx, max_ny - min_ny)]
                features['nose_for_pore'] = img[min_ny:max_ny, min_nx:max_nx]
                nose_for_pore = np.array(points['nose_for_pore'])
                if pore_rois is not None:
                    pore_rois = np.vstack((pore_rois, nose_for_pore))
                else:
                    pore_rois = nose_for_pore
                cv2.rectangle(overlay, (min_nx, min_ny), (max_nx, max_ny), (255, 255, 0), 2)
            elif feature == 'mouth':
                pass


        # 미간 추가
        if any(n >= 0 for n in [glabella_min_x, glabella_min_y, glabella_max_x, glabella_max_y]):
            glabella_min_y = glabella_mid_y - int((glabella_max_y - glabella_mid_y) * 1.3)
            points['glabella'] = [(glabella_min_x, glabella_min_y, glabella_max_x - glabella_min_x, glabella_max_y - glabella_min_y)]
            features['glabella'] = img[glabella_min_y:glabella_max_y, glabella_min_x:glabella_max_x]
            glabella = np.array(points['glabella'])
            if wrinkle_rois is not None:
                wrinkle_rois = np.vstack((wrinkle_rois, glabella))
            else:
                wrinkle_rois = glabella
            cv2.rectangle(overlay, (glabella_min_x, glabella_min_y), (glabella_max_x, glabella_max_y), (255, 0, 255), 2)

        # 오른쪽 볼 추가
        if any(n >= 0 for n in [right_cheek_min_x, right_cheek_min_y, right_cheek_max_x, right_cheek_max_y]):
            points['right_cheek'] = [(right_cheek_min_x, right_cheek_min_y, right_cheek_max_x - right_cheek_min_x,
                                      right_cheek_max_y - right_cheek_min_y)]
            features['right_cheek'] = img[right_cheek_min_y:right_cheek_max_y, right_cheek_min_x:right_cheek_max_x]
            right_cheek = np.array(points['right_cheek'])
            if pore_rois is not None:
                pore_rois = np.vstack((pore_rois, right_cheek))
            else:
                pore_rois = right_cheek
            cv2.rectangle(overlay, (right_cheek_min_x, right_cheek_min_y), (right_cheek_max_x, right_cheek_max_y), (255, 0, 255), 2)

        # 왼쪽 볼 추가
        if any(n >= 0 for n in [left_cheek_min_x, left_cheek_min_y, left_cheek_max_x, left_cheek_max_y]):
            points['left_cheek'] = [(left_cheek_min_x, left_cheek_min_y, left_cheek_max_x - left_cheek_min_x,
                                      left_cheek_max_y - left_cheek_min_y)]
            features['left_cheek'] = img[left_cheek_min_y:left_cheek_max_y, left_cheek_min_x:left_cheek_max_x]
            left_cheek = np.array(points['left_cheek'])
            if pore_rois is not None:
                pore_rois = np.vstack((pore_rois, left_cheek))
            else:
                pore_rois = left_cheek
            cv2.rectangle(overlay, (left_cheek_min_x, left_cheek_min_y), (left_cheek_max_x, left_cheek_max_y), (255, 0, 255), 2)

        # 눈, 코, 입 제거한 이미지 추가
        if any(n >= 0 for n in [face_min_x, face_min_y, face_max_x, face_max_y]):
            face_pts = points['jaw']
            for key, val in points.items():
                if key.endswith('_eye'):
                    skin = self.remove_eyes(removed_landmarks_img, landmarks[key], face_pts)
                elif key == 'nose': #'nose_for_pore'
                    skin = self.remove_nose(removed_landmarks_img, landmarks[key], face_pts)
                elif key == 'mouth':
                    skin = self.remove_mouth(removed_landmarks_img, landmarks[key], face_pts)
            features['skin_roi'] = skin

        # 모공 분석용 이미지들을 하나의 이미지로 만듦. Vertical append
        if len(pore_rois) > 0:
            pore_height, pore_width = self.calc_appended_image_size(np.array(pore_rois))
            # background_color = np.array([255, 255, 255], np.uint8)
            pore_img = np.full((pore_height, pore_width, 3), 255, dtype=np.uint8)
            prev_height = 0
            for i, pts in enumerate(pore_rois):
                temp_x, temp_y, temp_w, temp_h = pts
                pore_img[prev_height:prev_height+temp_h, 0:temp_w] = img[temp_y:temp_y+temp_h, temp_x:temp_x+temp_w].copy()
                prev_height += temp_h
            features['pore_roi'] = pore_img

        # 주름 분석용 이미지들을 하나의 이미지로 만듦. Vertical append
        if len(wrinkle_rois) > 0:
            wrinkle_height, wrinkle_width = self.calc_appended_image_size(np.array(wrinkle_rois))
            # background_color = np.array([255, 255, 255], np.uint8)
            wrinkle_img = np.full((wrinkle_height, wrinkle_width, 3), 255, dtype=np.uint8)
            prev_height = 0
            for i, pts in enumerate(wrinkle_rois):
                temp_x, temp_y, temp_w, temp_h = pts
                wrinkle_img[prev_height:prev_height+temp_h, 0:temp_w] = img[temp_y:temp_y+temp_h, temp_x:temp_x+temp_w].copy()
                prev_height += temp_h
            features['wrinkle_roi'] = wrinkle_img

        if visible:
            cv2.imshow('Features', overlay)
            if skin is not None:
                cv2.imshow('Skin', skin)
            cv2.waitKey()
            cv2.destroyAllWindows()

        return features, points


    def range_from_landmark(self, landmark):
        # print(landmark)
        min_xy = landmark.min(axis=0)
        min_x = min_xy[0, 0]
        min_y = min_xy[0, 1]
        max_xy = landmark.max(axis=0)
        max_x = max_xy[0, 0]
        max_y = max_xy[0, 1]

        return min_x, min_y, max_x, max_y


def average_color(img):
    b, g, r = cv2.split(img)

    avg_b = b.mean()
    avg_g = g.mean()
    avg_r = r.mean()

    return (avg_b, avg_g, avg_r)


def change_key(dictionary, mapped_key):
    return {mapped_key[key]: value for key, value in dictionary.items()}


def get_random_normal(mean=0, stddev=0.1, size=1, min_val=0, max_val=100):
    rand_val = int((max_val-min_val) / 2) + np.random.normal(mean, stddev, size) * max_val
    underflow_mask = rand_val < min_val
    overflow_mask = rand_val > max_val
    if np.max(underflow_mask):
        for i, is_underflow in enumerate(underflow_mask):
            rand_val[i] = min_val
    if np.max(overflow_mask):
        for i, is_overflow in enumerate(overflow_mask):
            rand_val[i] = max_val
        rand_val = min(rand_val, max_val)
    return rand_val

def get_score_data(erythema=-1.0, emotion=-1.0, pigmentation=-1.0, pore=-1.0, wrinkle=-1.0):
    # 총점 계산(산술 평균) 시에 적용할 각 요소별 가중치, 각 가중치의 총합은 5가 되도록 유지해야함.
    WEIGHT_ERYTHEMA = 1.0
    WEIGHT_EMOTION = 1.0
    WEIGHT_PIGMENTATION = 1.0
    WEIGHT_PORE = 1.0
    WEIGHT_WRINKLE = 1.0
    # 효율적인 1-D vector (array_like) 계산을 위해 numpy array로 생성.
    weights = np.array([WEIGHT_ERYTHEMA, WEIGHT_EMOTION, WEIGHT_PIGMENTATION, WEIGHT_PORE, WEIGHT_WRINKLE])

    # 각 요소별 측정값에 대한 입력값
    points = [erythema, emotion, pigmentation, pore, wrinkle]

    # 요소에 대한 정보를 매개변수로 입력 받지 못한 경우 해당 점수를 임의값(범위: 0 ~ 1)으로 지정 후 numpy array로 생성.
    scores = np.array([round(score, 2) if score > 0 else round(get_random_normal()[0], 2) for score in points])
    # 가중치를 적용한 평균 계산.
    score_total = np.average(scores * weights)
    score_total = round(score_total, 2)

    return {
            'score_erythema': scores[0],
            'score_emotion': scores[1],
            'score_pigmentation': scores[2],
            'score_pore': scores[3],
            'score_wrinkle': scores[4],
            'score_total': score_total,
            }
