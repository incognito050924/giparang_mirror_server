import time
import os
import copy
from django.shortcuts import render
from rest_framework import viewsets, filters, status, generics, views
from rest_framework.decorators import api_view
from django.http import HttpResponse, JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django_filters.rest_framework import DjangoFilterBackend
from django.core.files.storage import FileSystemStorage
from django.conf import settings
from .models import User,  SkinData
from .serializers import UserSerializer, ResultSerializer, MeasuredSerializer, SkinDataSerializer
from .filters import SkinDataFilter
from datetime import date
from .services.image_processor import bytes2opencv_img, resize_image, bgr2rgb
from .services.analysis import CascadeDetector, LandmarkDetector, Analyzer, get_score_data, NoFaces, TooManyFaces
from .ml.predictor import predict_emotion
import numpy as np
from django.core.files.base import ContentFile
from django.core.files.uploadedfile import InMemoryUploadedFile
from PIL import Image
from io import BytesIO


# Create your views here.
class UserViewSet(viewsets.ModelViewSet):
    """

    """
    queryset = User.objects.all()
    serializer_class = UserSerializer
    filter_backends = (filters.SearchFilter,)
    search_fields = ('user_id',)


class ResultViewSet(viewsets.ModelViewSet):
    """

    """
    queryset = SkinData.objects.all()
    serializer_class = ResultSerializer
    filter_class = SkinDataFilter


class MeasuredViewSet(views.APIView) :
    """

    """
    queryset = SkinData.objects.all()
    serializer_class = MeasuredSerializer

    # def post(self, request, format='jpeg'):
    #     image = request.FILES['image']
    #     print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@\n",image)
    #     print(type(image))



# ModelViewSet은 `list`와 `create`, `retrieve`, `update`, 'destroy` 기능을 자동으로 지원합니다
class SkinDataViewSet(viewsets.ModelViewSet):
    """
    SkinData에 대한 뷰셋.
    """
    queryset = SkinData.objects.all()
    serializer_class = SkinDataSerializer

def to_json(query_set):
    for data in query_set:
        dictionary = {data.as_dict()}
    return dictionary

# @csrf_exempt
# def get_data_by_user(request, pk):
#     try:
#         dataset = User.objects.get(pk=pk).measureddata_set.all()
#         err_msg = None
#     except User.DoesNotExist:
#         dataset = None
#         err_msg = 'User data does not exist.'
#     except SkinStatus.DoesNotExist:
#         dataset = None
#         err_msg = 'Skin status does not exist.'
#
#     if request.method == 'GET':
#         serializer = SkinStatusSerializer(dataset, many=True, context={'request': request}, error_messages=err_msg)
#         return JsonResponse(serializer.data, safe=False)

@api_view(['GET', 'POST'])
def result_list(request):
    """
    측정 데이터 분석 후 분석데이터 저장(POST method) 및 점수리스트 조회(GET method)
    """
    # When server received request by GET Method, then return all result data.
    if request.method == 'GET':
        status_list = SkinData.objects.filter(
        measured_at__startswith=date.today()).order_by('-measured_at')[:1]
        serializer = ResultSerializer(status_list, many=True, context={'request': request})
        return JsonResponse(serializer.data, safe=False)

    # When server received request by POST Method, then create result data.
    elif request.method == 'POST':
        serializer = MeasuredSerializer(data=request.data)
        if serializer.is_valid():
            serializer.save()
            return JsonResponse(serializer.data, status=status.HTTP_201_CREATED)
        return JsonResponse(serializer.errors, status=status.HTTP_400_BAD_REQUEST)


@api_view(['GET', 'POST'])
def result_list_by_year_month(request, year, month):
    """
    년/월에 해당하는 데이터를 리턴한다.
    """
    # status_list = SkinData.objects.filter(measured_at__year=year, measured_at__month=month).order_by('-measured_at')
    # for skindata in status_list:
    #     print(skindata.measured_at.date())
    status_list = []
    measured_dates = SkinData.objects.dates('measured_at', 'day', order='ASC')
    for measured_date in measured_dates:
        if measured_date.year == int(year) and measured_date.month == int(month):
            status_list.extend(list(SkinData.objects.filter(measured_at__startswith=measured_date).order_by('-measured_at')[:1]))
    serializer = ResultSerializer(status_list, many=True, context={'request': request})
    return JsonResponse(serializer.data, safe=False)


@api_view(['GET', 'POST'])
def result_list_by_year_month_day(request, year, month, day):
    """
    년/월/일에 해당하는 데이터를 리턴한다.
    """
    # status_list = SkinData.objects.filter(
    #     measured_at__year=year,
    #     measured_at__month=month,
    #     measured_at__day=day).order_by('-measured_at')
    status_list = []
    today_status = SkinData.objects.order_by('-measured_at').annotate()[:1]
    status_list.extend(list(today_status))

    status = SkinData.objects.filter(
        measured_at__startswith=date(int(year), int(month), int(day))).order_by('-measured_at')[:1]
    status_list.extend(list(status))

#    today_status['is_today'] = 'Y'
    serializer = ResultSerializer(status_list, many=True, context={'request': request})
    return JsonResponse(serializer.data, safe=False)


# @api_view(['GET', 'PUT', 'DELETE'])
# def skin_data_detail(request, pk):
#     """
#     id를 통한 피부 데이터 조회, 수정, 삭제
#     """
#     try:
#         result = SkinData.objects.get(pk=pk)
#     except SkinData.DoesNotExist:
#         return JsonResponse(status=status.HTTP_404_NOT_FOUND)
#
#     # Find data by id(=pk)
#     if request.method == 'GET':
#         serializer = SkinDataSerializer(result)
#         return JsonResponse(data=serializer.data)
#
#     elif request.method == 'PUT':
#         serializer = SkinDataSerializer(result, data=request.data)
#         if serializer.is_valid():
#             serializer.save()
#             return JsonResponse(serializer.data)
#         return JsonResponse(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
#
#     elif request.method == 'DELETE':
#         result.delete()
#         return JsonResponse(status=status.HTTP_204_NO_CONTENT)

@api_view(['POST'])
def test(request):
    """
    측정 데이터 분석 후 분석데이터 저장(POST method) 및 점수리스트 조회(GET method)
    """
    start_ms = int(round(time.time() * 1000))
    data = request.data
    temp_img = copy.deepcopy(request.FILES['image'])
    img = bytes2opencv_img(temp_img)

    # 표정 예측 모델
    gray_face = CascadeDetector().detect_face(img, use_gray=True, visible=False)
    face = resize_image(gray_face, (299, 299, 1))
    face = np.reshape(face, (299, 299, 1))
    emotion, score = predict_emotion(np.expand_dims(face, 0), text_label=True)
    emotion_data = {'emotion': emotion, 'score_emotion': score}
    try:
        # 피부 분석 모듈
        features, points = LandmarkDetector().detect_facial_feature(img, visible=False)
        pore_img = features['pore_roi']
        wrinkle_img = features['wrinkle_roi']
        skin_img = features['skin_roi']

        set_visible = False
        analyzer = Analyzer()
        result_data = analyzer.analyze_emotion(emotion_data)
        result_data = analyzer.analyze_erythema(skin_img, result_data=result_data, visible=set_visible)
        result_data = analyzer.analyze_pore(pore_img, result_data=result_data, visible=set_visible)
        result_data = analyzer.analyze_pigmentation(skin_img, result_data=result_data, visible=set_visible)
        result_data = analyzer.analyze_wrinkle(wrinkle_img, result_data=result_data, visible=set_visible)
        result_data = analyzer.calc_total_score(result_data)

        # score_dict = get_score_data()
        data.update(result_data)
        print(data)
        serializer = SkinDataSerializer(data=data)
        if serializer.is_valid():
            # print(serializer.validated_data['image'])
            serializer.save()
            # 응답시간
            print(int(round(time.time() * 1000)) - start_ms, 'milliseconds')
            return JsonResponse(serializer.data, status=status.HTTP_201_CREATED)
        return JsonResponse(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
    except NoFaces as no_face_err:
        print('No faces', no_face_err)
        error = {'error_msg': '인식된 얼굴이 없습니다.'}
        return JsonResponse(error, status=status.HTTP_500_INTERNAL_SERVER_ERROR, safe=False)
    except TooManyFaces as too_many_err:
        print('Too many faces', too_many_err)
        error = {'error_msg': '너무 많은 얼굴이 인식되었습니다.'}
        return JsonResponse(error, status=status.HTTP_500_INTERNAL_SERVER_ERROR, safe=False)
    # except Error as e:
    #     print(repr(e))
    #     return JsonResponse(serializer.errors, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

        # return JsonResponse(data)
