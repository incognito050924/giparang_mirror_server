from django.shortcuts import render
from rest_framework import viewsets, filters, status, generics
from rest_framework.decorators import api_view
from django.http import HttpResponse, JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django_filters.rest_framework import DjangoFilterBackend
from .models import User,  SkinData
from .serializers import UserSerializer, ResultSerializer, MeasuredSerializer, SkinDataSerializer
from .filters import SkinDataFilter
from datetime import date, datetime


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


class MeasuredViewSet(viewsets.ModelViewSet):
    """

    """
    queryset = SkinData.objects.all()
    serializer_class = MeasuredSerializer


# ModelViewSet은 `list`와 `create`, `retrieve`, `update`, 'destroy` 기능을 자동으로 지원합니다
class SkinDataViewSet(viewsets.ModelViewSet):
    """
    SkinData에 대한 뷰셋.
    """
    queryset = SkinData.objects.all()
    serializer_class = SkinDataSerializer


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
        status_list = SkinData.objects.all()
        serializer = ResultSerializer(status_list, many=True, context={'request': request})
        return JsonResponse(serializer.data)

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
    status_list = SkinData.objects.none()
    measured_dates = SkinData.objects.dates('measured_at', 'day', order='DESC')
    for measured_date in measured_dates:
        if measured_date.year == int(year) and measured_date.month == int(month):
            status_list = status_list.union(SkinData.objects.filter(measured_at__startswith=measured_date).order_by('-measured_at')[:1])
    serializer = ResultSerializer(status_list, many=True, context={'request': request})
    return JsonResponse(serializer.data, safe=False)


@api_view(['GET', 'POST'])
def result_list_by_year_month_day(request, year, month, day):
    """
    년/월/일에 해당하는 데이터를 리턴한다.
    """
    status_list = SkinData.objects.filter(
        measured_at__year=year,
        measured_at__month=month,
        measured_at__day=day).order_by('-measured_at')[:1]
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
