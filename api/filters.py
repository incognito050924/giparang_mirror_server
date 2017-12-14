import django_filters
import datetime
import time
from django import forms
from .models import SkinData


class DatetimeRangeField(django_filters.RangeFilter):
    def __init__(self, *args, **kwargs):
        fields = (
            forms.DateTimeField(),
            forms.DateTimeField())
        super(DatetimeRangeField, self).__init__(fields, *args, **kwargs)

    def compress(self, data_list):
        if data_list:
            start_datetime, stop_datetime = data_list
            if start_datetime:
                start_datetime = datetime.combine(start_datetime, time.min)
                if stop_datetime:
                    stop_datetime = datetime.combine(stop_datetime, time.max)
                    return slice(start_datetime, stop_datetime)
        return None


class DatetimeFromToRangeFilter(django_filters.RangeFilter):
    field_class = DatetimeRangeField


class SkinDataFilter(django_filters.FilterSet):
    start_time = DatetimeFromToRangeFilter()

    class Meta:
        model = SkinData
        fields = ['measured_at']
