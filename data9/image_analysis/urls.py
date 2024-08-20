from django.urls import path
from . import views
from django.conf import settings
from django.conf.urls.static import static


urlpatterns = [
    path('', views.index, name='index'),
    path('process/', views.process_file, name='process'),
    path('model_LeNet/', views.model_LeNet, name='model_LeNet'),
    path('model_LeNet_tuned/', views.model_LeNet_tuned, name='model_LeNet_tuned'),
    path('model_VGG16/', views.model_VGG16, name='model_VGG16'),
    path('model_CNN/', views.model_CNN, name='model_CNN'),
    path('download_model/<str:model_type>/', views.download_model, name='download_model'),
] + static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
