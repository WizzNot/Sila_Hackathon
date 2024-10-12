from django.shortcuts import render
from django.conf import settings
from django.core.files.storage import FileSystemStorage
from PIL import Image
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
        predicted_images = []
        for image in images:
            unique_filename = str(uuid.uuid4()) + os.path.splitext(image.name)[-1]
            fs = FileSystemStorage()
            fs.save("images/" + unique_filename, image)
            image_names.append(unique_filename)
        results = model.predict(os.path.join(settings.BASE_DIR, "images"))
        for i in image_names:
            delete_image(i)
        for i, r in enumerate(results):
            name = image_names[i]
            print(name)
            r.save(filename=os.path.join("src", "predicted_images", name))
            predicted_images.append(name)
        return render(request, 'homepage/upload_success.html', {"images": predicted_images})
    return render(request, 'homepage/upload_fail.html')