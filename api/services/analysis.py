import random
import numpy as np
import cv2
import dlib
import math
from network import client


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
        self.erythema = client.request(request_data)

    def extract_pore(self, img,
                     tv_1=client.TrackBarValue.NORMAL,
                     tv_2=client.TrackBarValue.NORMAL,
                     tv_3=client.TrackBarValue.NORMAL,
                     tv_4=client.TrackBarValue.NORMAL):

        request_data = client.serialize_req_param_data(client.SkinFeature.PORE, img, tv_1, tv_2, tv_3, tv_4, False)
        self.pore = client.request(request_data)

    def extract_pigmentation(self, img,
                     tv_1=client.TrackBarValue.NORMAL,
                     tv_2=client.TrackBarValue.NORMAL,
                     tv_3=client.TrackBarValue.NORMAL,
                     tv_4=client.TrackBarValue.NORMAL):

        request_data = client.serialize_req_param_data(client.SkinFeature.PIGMENTATION, img, tv_1, tv_2, tv_3, tv_4, False)
        self.pigmentation = client.request(request_data)

    def extract_wrinkle(self, img,
                     tv_1=client.TrackBarValue.NORMAL,
                     tv_2=client.TrackBarValue.NORMAL,
                     tv_3=client.TrackBarValue.NORMAL,
                     tv_4=client.TrackBarValue.NORMAL):

        request_data = client.serialize_req_param_data(client.SkinFeature.WRINKLE, img, tv_1, tv_2, tv_3, tv_4, False)
        self.wrinkle = client.request(request_data)

    def extract_all(self, img,
                     tv_1=client.TrackBarValue.NORMAL,
                     tv_2=client.TrackBarValue.NORMAL,
                     tv_3=client.TrackBarValue.NORMAL,
                     tv_4=client.TrackBarValue.NORMAL):
        self.extract_erythema(img, tv_1, tv_2, tv_3, tv_4)
        self.extract_pore(img, tv_1, tv_2, tv_3, tv_4)
        self.extract_pigmentation(img, tv_1, tv_2, tv_3, tv_4)
        self.extract_wrinkle(img, tv_1, tv_2, tv_3, tv_4)

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
    def analyze_erythema(self, erythema_data):
        num_erythema = erythema_data['Count']
        avg_area = erythema_data['Area']
        avg_darkness = erythema_data['Darkness']
        print('Erythema[ %d, %d, %d ]' % (num_erythema, avg_area, avg_darkness))

    def analyze_pore(self, pore_data):
        num_pore = pore_data['Count']
        print('Pore[ %d ]' % (num_pore))

    def analyze_pigmentation(self, pigmentation_data):
        num_pigmentation = pigmentation_data['Count']
        avg_area = pigmentation_data['Area']
        avg_darkness = pigmentation_data['Darkness']
        print('Pigmentation[ %d, %d, %d ]' % (num_pigmentation, avg_area, avg_darkness))

    def analyze_wrinkle(self, wrinkle_data):
        num_wrinkle = wrinkle_data['Count']
        avg_area = wrinkle_data['Area']
        avg_darkness = wrinkle_data['Darkness']
        pitch = wrinkle_data['Pitch']
        length = wrinkle_data['Length']
        print('Wrinkle[ %d, %d, %d, %d, %d ]' % (num_wrinkle, avg_area, avg_darkness, pitch, length))


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
                    if k_mean < 1: k_mean = 1
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


    def detect_facial_feature(self, img, visible=False):
        face_cascade = cv2.CascadeClassifier('cascades/frontalFace.xml')
        eye_cascade = cv2.CascadeClassifier('cascades/Eyes_cascade.xml')
        nose_cascade = cv2.CascadeClassifier('cascades/Nose_cascade.xml')
        mouth_cascade = cv2.CascadeClassifier('cascades/Mouth_cascade.xml')

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
        PREDICTIOR_PATH = 'shape_predictor_68_face_landmarks.dat'
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
            cv2.putText(img, str(idx), pos, fontFace=cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, fontScale=0.4, color=(0, 0, 255))
            cv2.circle(img, pos, 3, color=(0, 255, 255))

        return overlay

    def eye_dilate(self, eyes, x_rate=1.6, y_rate=2):
        result = eyes

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

    def eye_brow_dilate(self, eye_brows, x_rate=1.1, y_rate=1.3):
        result = eye_brows

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

    def nose_dilate(self, nose, x_rate=1.3, y_rate=1):
        result = nose

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

    def mouth_dilate(self, mouth, x_rate=1.2, y_rate=1.2):
        result = mouth

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

    def remove_landmarks_from_img(self, img, landmarks, color=(0,0,0),
                      use_feature=['jaw', 'right_eyebrow', 'left_eyebrow', 'nose', 'right_eye', 'left_eye', 'mouth']):
        # create two copies of the input image -- one for the
        # overlay and one for the final output image
        overlay = img.copy()

        facial_landmarks = self.matrix2dict(landmarks)
        for i, feature in enumerate(use_feature):
            pts = facial_landmarks[feature]
            if feature == 'jaw':
                for idx in range(1, len(pts)):
                    ptA = (pts[idx - 1][0, 0], pts[idx - 1][0, 1])
                    ptB = (pts[idx][0, 0], pts[idx][0, 1])
                    cv2.line(overlay, ptA, ptB, color, 2)
                    cv2.circle(overlay, ptB, 3, color=(0, 255, 255))

            else:
                if feature == 'right_eye' or feature == 'left_eye':
                    pts = self.eye_dilate(facial_landmarks[feature])
                elif feature == 'right_eyebrow' or feature == 'left_eyebrow':
                    pts = self.eye_brow_dilate(facial_landmarks[feature])
                elif feature == 'mouth':
                    pts = self.mouth_dilate(facial_landmarks[feature])
                elif feature == 'nose':
                    pts = self.nose_dilate(facial_landmarks[feature], x_rate=1.5)
                hull = cv2.convexHull(pts)
                cv2.drawContours(overlay, [hull], -1, color, -1)

                for idx in range(len(pts)):
                    pos = (pts[idx][0, 0], pts[idx][0, 1])
                    cv2.circle(overlay, pos, 3, color=(0, 255, 255))

        landmark_area_min_x, landmark_area_min_y, landmark_area_max_x, landmark_area_max_y = self.range_from_landmark(landmarks)
        overlay = overlay[landmark_area_min_y:landmark_area_max_y, landmark_area_min_x:landmark_area_max_x]

        cv2.imshow('Remove_Landmarks', overlay)
        cv2.waitKey()
        cv2.destroyAllWindows()

        return overlay

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

        glabella_min_x, glabella_min_y, glabella_max_x, glabella_max_y, glabella_mid_y = (-1, -1, -1, -1, 0)

        for i, feature in enumerate(use_feature):
            min_x, min_y, max_x, max_y = self.range_from_landmark(landmarks[feature])
            features[feature] = img[min_y:max_y, min_x:max_x]
            cv2.rectangle(overlay, (min_x, min_y), (max_x, max_y), colors[i], 2)

            if feature == 'right_eye':
                side_min_x = min_x - int((max_x - min_x) * 0.7)
                side_max_x = min_x
                features['right_eye_side'] = img[min_y:max_y, side_min_x:side_max_x]
                cv2.rectangle(overlay, (side_min_x, min_y), (side_max_x, max_y), (255, 255, 0), 2)
            elif feature == 'left_eye':
                side_min_x = max_x
                side_max_x = max_x + int((max_x - min_x) * 0.7)
                features['left_eye_side'] = img[min_y:max_y, side_min_x:side_max_x]
                cv2.rectangle(overlay, (side_min_x, min_y), (side_max_x, max_y), (255, 255, 0), 2)
            elif feature == 'right_eyebrow':
                glabella_min_x = max_x
                glabella_mid_y = max(glabella_mid_y, max_y - int((max_y - min_y) / 2))
            elif feature == 'left_eyebrow':
                glabella_max_x = min_x
                glabella_mid_y = max(glabella_mid_y, max_y - int((max_y - min_y) / 2))
            elif feature == 'nose':
                glabella_min_y = glabella_mid_y - (min_y - glabella_mid_y)
                glabella_max_y = min_y

        # 미간 추가
        if any(n >= 0 for n in [glabella_min_x, glabella_min_y, glabella_max_x, glabella_max_y]):
            glabella_min_y = glabella_mid_y - int((glabella_max_y - glabella_mid_y) * 1.3)
            features['glabella'] = img[glabella_min_y:glabella_max_y, glabella_min_x:glabella_max_x]
            cv2.rectangle(overlay, (glabella_min_x, glabella_min_y), (glabella_max_x, glabella_max_y), (255, 0, 255), 2)

        if visible:
            cv2.imshow('Features', overlay)
            cv2.waitKey()
            cv2.destroyAllWindows()

        return features


    def range_from_landmark(self, landmark):
        # print(landmark)
        min_xy = landmark.min(axis=0)
        min_x = min_xy[0, 0]
        min_y = min_xy[0, 1]
        max_xy = landmark.max(axis=0)
        max_x = max_xy[0, 0]
        max_y = max_xy[0, 1]

        return (min_x, min_y, max_x, max_y)


def average_color(img):
    b, g, r = cv2.split(img)

    avg_b = b.mean()
    avg_g = g.mean()
    avg_r = r.mean()

    return (avg_b, avg_g, avg_r)


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
    scores = np.array([round(score, 2) if score > 0 else round(random.random() * 100, 2) for score in points])
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
