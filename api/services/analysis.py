import random
import numpy as np
from .image_processor import *


class Extractor:
    """
    이미지에서 각 요소를 추출할 수 있는 함수를 제공한다.
    """
    pass


class Analyzer:
    """
    추출된 요소의 갯수 등을 이용하여 점수화하는 함수를 제공한다.
    """
    pass


class Detector:
    """
    각 요소 추출을 위해 ROI를 찾는 기능을 제공한다.
    """
    pass


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
