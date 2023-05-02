from django.shortcuts import render, HttpResponse
from django.views.decorators.csrf import csrf_exempt
from predict.models import *
from predict.utils import *

# Create your views here.
def index(request):
    return render(request, "predict/base.html")

@csrf_exempt
def prediction(request):
    if request.method == 'POST':
        music_file = request.FILES['doc']
        try:
            file = File_model.objects.create(upload=music_file)
            file.save()
        except Exception as e:
            raise e
        audio_file = File_model.objects.last().upload
        audio_file_path = audio_file.path
        audio_file_name = audio_file.name
        csv_file_path = extract_features(audio_file_path, audio_file_name)
        genre_prediction = predictor(csv_file_path)
        return render(request, "predict/result.html",{
            "genre": genre_prediction.capitalize()
        })
        #print(genre_prediction)
    else:
        return HttpResponse(status=400)