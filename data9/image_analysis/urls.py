from django.urls import path
from . import views
from django.conf import settings
from django.conf.urls.static import static


urlpatterns = [
    path('', views.index, name='index'),
    path('process/', views.process_file, name='process'),
    path('team_info/', views.team_info, name='team_info'),
    path('model_alexnet/', views.model_alexnet, name='model_alexnet'),
    path('model_LeNet_tuned/', views.model_LeNet_tuned, name='model_LeNet_tuned'),
    path('model_VGG16/', views.model_VGG16, name='model_VGG16'),
    path('model_mobnet/', views.model_mobnet, name='model_mobnet'),
    path('download_model/<str:model_type>/', views.download_model, name='download_model'),
] + static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
