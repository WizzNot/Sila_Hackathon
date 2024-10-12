from django.shortcuts import render
from django.conf import settings
from django.core.files.storage import FileSystemStorage
import uuid
import os

model = settings.MODEL


def delete_image(filename):
    filepath = os.path.join(settings.BASE_DIR, "images", filename)
    if os.path.exists(filepath):
        os.remove(filepath)
        print("deleted")
    else:
        print("error")


def homepage(request):
    template = "homepage/home.html"
    return render(request, template_name=template)


def upload(request):
    if request.method == 'POST' and request.FILES.getlist('images'):
        images = request.FILES.getlist('images')
        image_names = []
        for image in images:
            unique_filename = str(uuid.uuid4()) + os.path.splitext(image.name)[-1]
            fs = FileSystemStorage()
            fs.save("images/" + unique_filename, image)
            image_names.append(unique_filename)
        results = model.predict(os.path.join(settings.BASE_DIR, "images"))
        for i in image_names:
            delete_image(i)
        return render(request, 'homepage/upload_success.html')
    return render(request, 'homepage/upload_fail.html')