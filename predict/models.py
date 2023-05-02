from django.db import models

class File_model(models.Model):
    upload = models.FileField(upload_to="uploads/")
