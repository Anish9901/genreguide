from django.urls import path

from . import views
""" app_name = "predict" """
urlpatterns = [
    path("", views.index, name="index"),
    path("prediction/",views.prediction, name="prediction")
]