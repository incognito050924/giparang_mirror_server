"""giparang_mirror_server URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/1.11/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  url(r'^$', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  url(r'^$', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.conf.urls import url, include
    2. Add a URL to urlpatterns:  url(r'^blog/', include('blog.urls'))
"""
from django.conf.urls import url, include, static
from django.conf import settings
from django.contrib import admin
from rest_framework import routers
from api import views

router = routers.DefaultRouter()
#router.register(r'analyze', views.MeasuredViewSet)
router.register(r'result', views.ResultViewSet)
router.register(r'skindata', views.SkinDataViewSet)

urlpatterns = [
    url(r'^admin/', admin.site.urls),
    url(r'^api-auth/', include('rest_framework.urls', namespace='rest_framework')),
    url(r'^api/', include(router.urls)),
    url(r'^api/analyze', views.result_list),
    #url(r'^api/result//$', views.result_list),
    url(r'^api/result/(?P<year>\d+)/(?P<month>\d+)/$', views.result_list_by_year_month),
    url(r'^api/result/(?P<year>\d+)/(?P<month>\d+)/(?P<day>\d+)/$', views.result_list_by_year_month_day),
    #url(r'^api/analyze/$', views.test),
    # url(r'^api/skindata/$', views.SkinDataViewSet.as_view({
    #     'get': 'list',
    #     'post': 'create'
    # })),
    # url(r'^api/skindata/(?P<pk>\d+)/$', views.SkinDataViewSet.as_view({
    #     'get': 'retrieve',
    #     'put': 'update',
    #     'patch': 'partial_update',
    #     'delete': 'destroy'
    # })),
]+static.static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)


"""
1. Analyze facial image. And return analyzed data.
2. Show history with measured date or due term.
    @dueto: ['latest', 'monthly' , 'year', 'select']
"""