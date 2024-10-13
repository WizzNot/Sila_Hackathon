from django.shortcuts import render
from django.conf import settings
from django.core.files.storage import FileSystemStorage
import pandas as pd
import json
from PIL import Image
import uuid
import os

model = settings.MODEL

d = {'царапины': 0, 'битые пиксели': 1, 'проблемы с клавишами': 2, 'замок': 3, 
     'отсутствует шуруп': 4, 'сколы': 5, 'специфический': 6}


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
            fs.save("learn_images/" + unique_filename, image)
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


def process_coordinates(request):
  if request.method == 'POST':
    data = json.loads(request.body)
    image = data.get('image')
    point1 = data.get('point1')
    point2 = data.get('point2')
    damageclass = data.get('damage_class')
    filename = data.get('image')
    filename = filename.split("/")[-1]
    x1, y1, x2, y2 = int(point1['x']), int(point1['y']), int(point2['x']), int(point2['y'])
    dx, dy = int(min(x1, x2)), int(max(y1, y2))
    width = int(abs(max(x1, x2) - dx))
    length = int(abs(dy - min(y2, y1)))
    df = pd.DataFrame(columns=['main_class','filename','x_left_bottom','y_left_bottom','length','width'])
    df.loc[len(df)] = [d[damageclass.lower()],filename,dx,dy,length,width]
    old_df = pd.read_csv(os.path.join(settings.BASE_DIR, "src", "data.csv"))
    combined = pd.concat([old_df, df], ignore_index=True)
    combined.to_csv(os.path.join(settings.BASE_DIR, "src", "data.csv"), index=False)
    

    return render(request, 'base.html', {'images': []}) # Возвращаем шаблон (или другой ответ)
  else:
    return render(request, 'base.html', {'images': []}) # Возвращаем шаблон (или другой ответ)