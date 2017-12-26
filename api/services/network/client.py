import socket
import io
import json
from scipy import misc
from enum import Enum


class SkinFeature(Enum):
    ERYTHEMA = 'ery'
    PORE = 'por'
    PIGMENTATION = 'pig'
    WRINKLE = 'wri'


class TrackBarValue(Enum):
    TIGHT = 3
    NORMAL = 2
    LOOSE = 1


def deserialize_resp_data(response_data):
    """
    서버로부터 받은 응답을(JSON 형식의 string -> bytes 데이터) dict으로 변환한다.
    :param response_data: byte(JSON string)
    :return: dict()
    """
    #return pickle.loads(response_data)
    return json.loads(response_data)


def serialize_req_param_data(feature_type, roi_cv_img,
              tv_1=TrackBarValue.NORMAL,
              tv_2=TrackBarValue.NORMAL,
              tv_3=TrackBarValue.NORMAL,
              tv_4=TrackBarValue.NORMAL,
              visible=False):
    """
    서버로 보낼 요청 데이터를 bytes로 변환한다.
    :param feature_type: Enum 타입 client.SkinFeature 중 하나의 값
    :param tv_1: Enum 타입 client.TrackBarValue 중 하나의 값
    :param tv_2: Enum 타입 client.TrackBarValue 중 하나의 값
    :param tv_3: Enum 타입 client.TrackBarValue 중 하나의 값
    :param tv_4: Enum 타입 client.TrackBarValue 중 하나의 값
    :param roi_cv_img: OpenCV-python의 이미지(numpy array)
    :param visible: 이미지 출력 여부 (소켓 서버 디버깅 용)
    :return: 요청에 대한 응답 dict()
    """
    # Type checking
    if not isinstance(feature_type, SkinFeature):
        raise TypeError('feature_type must be an instance of SkinFeature Enum')

    if not isinstance(tv_1, TrackBarValue):
        raise TypeError('tv_1 must be an instance of TrackBarValue Enum')

    if not isinstance(tv_2, TrackBarValue):
        raise TypeError('tv_2 must be an instance of TrackBarValue Enum')

    if not isinstance(tv_3, TrackBarValue):
        raise TypeError('tv_3 must be an instance of TrackBarValue Enum')

    if not isinstance(tv_4, TrackBarValue):
        raise TypeError('tv_4 must be an instance of TrackBarValue Enum')

    data = dict()
    data['type'] = feature_type.value
    # data['ROI'] = roi
    data['tv_1'] = tv_1.value
    data['tv_2'] = tv_2.value
    data['tv_3'] = tv_3.value
    data['tv_4'] = tv_4.value
    data['visible'] = int(visible)

    # data serialization to str
    json_str = bytes(json.dumps(data), 'utf-8')

    # Append RoI Data
    # Request Data format: byte[78] <- json_str(Parameters) + Image bytes
    json_str += image_to_bytes(roi_cv_img)

    return json_str


def image_to_bytes(img):
    """
    OpenCV-python에서 이미지 (numpy array)를 Bytes로 만든다.
    :param img: Image(numpy array)
    :return: Image(Bytes)
    """
    # Convert BGR(opencv) to RGB
    img_rgb = img[..., ::-1]
    output = io.BytesIO()
    misc.toimage(img_rgb).save(output, format='JPEG')

    return output.getvalue()


def request(send_data):
    """
    서버로 요청을 보내고 응답을 리턴한다.
    :param send_data: Serialized Request Data > Bytes or byte[]
    :return: Deserialized Response Data > dict()
    """
    host, port = "127.0.0.1", 34567
    # Create a socket (SOCK_STREAM means a TCP socket)
    client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

    try:
        # Connect to server and send data
        client.connect((host, port))
        client.sendall(send_data)

        # Receive data from server
        data = client.recv(1024)

        # Deserialize received JSON data
        data = deserialize_resp_data(data)

    finally:
        client.close()

    return data
