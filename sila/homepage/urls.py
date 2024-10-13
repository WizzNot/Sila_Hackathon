from django.urls import path
from homepage import views

app_name = "homepage"

urlpatterns = [
    path("", views.homepage, name="home"),
    path("upload/", views.upload, name="upload"),
    path("process_coordinates/", views.process_coordinates, name="process_coordinates"),
    path("modelfit/", views.learn, name="learn"),
]