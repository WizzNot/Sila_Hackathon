from django.shortcuts import render
from django.conf import settings
from django.core.files.storage import FileSystemStorage
import pandas as pd
import json
from PIL import Image
import uuid
import os

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
        results = settings.MODEL.predict(os.path.join(settings.BASE_DIR, "images"))
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


# MODEL_FITTING

def learn(request):
    import cv2
    import os
    import numpy as np
    import pandas as pd
    from django.conf import settings

    df = pd.read_csv(os.path.join(settings.BASE_DIR, "src", "data.csv"), delimiter=",")

    input_folder = os.path.join(settings.BASE_DIR, 'learn_images')
    output_folder = os.path.join(settings.BASE_DIR, 'src', 'dataset', 'train')
    output_images_folder = os.path.join(output_folder, 'images')
    output_labels_folder = os.path.join(output_folder, 'labels')

    resize_const = 640

    os.makedirs(output_images_folder, exist_ok=True)
    os.makedirs(output_labels_folder, exist_ok=True)

    def normalize_coordinates(x1, y1, x2, y2, img_width, img_height):
        x_center = (x1 + x2) / 2 / img_width
        y_center = (y1 + y2) / 2 / img_height
        width = abs(x2 - x1) / img_width
        height = abs(y2 - y1) / img_height
        return x_center, y_center, width, height

    for filename in os.listdir(input_folder):
        image_path = input_folder + "/" + filename
        image = cv2.imread(image_path)
        if len(df[df['filename']==filename].values) != 0:
            resized_image = cv2.resize(image, (resize_const, resize_const))
            x1, y1, length, w = (df[df['filename']==filename].values[0][2:6])
            
            x2 = x1 + length
            y2 = y1 - w
            # # Вырезаем область изображения
            # cropped_image = image[y2:y1, x1:x2]
            # print(len(cropped_image))

            # # Сохраняем или отображаем вырезанное изображение
            # cv2.imshow('Cropped Image', cropped_image)
            # cv2.waitKey(0)  # Ожидаем нажатия клавиши
            x_center, y_center, width, height = normalize_coordinates(x1, y1, x2, y2, image.shape[1], image.shape[0])

            # Сохранение изображения
            output_image_path = os.path.join(output_images_folder, filename)
            cv2.imwrite(output_image_path, resized_image)
            
            # Сохранение 
            label_filename = os.path.splitext(filename)[0] + '.txt'
            label_file_path = os.path.join(output_labels_folder, label_filename)
            with open(label_file_path, 'w') as f:
                f.write(f"{df[df['filename']==filename].values[0][0]} {x_center} {y_center} {width} {height}\n")
            print(f"Processed and saved {filename}")

    # SECOND PART

    import os
    import shutil
    import random

    # Параметры для разделения данных
    test_percent = 0.2  # Процент данных для тестирования
    valid_percent = 0.1  # Процент данных для проверки

    # Путь к папке с данными
    dataset_path = os.path.join(settings.BASE_DIR, 'src', 'dataset')
    train_images_path = os.path.join(dataset_path, 'train', 'images')
    train_labels_path = os.path.join(dataset_path, 'train', 'labels')
    valid_images_path = os.path.join(dataset_path, 'valid', 'images')
    valid_labels_path = os.path.join(dataset_path, 'valid', 'labels')
    test_images_path = os.path.join(dataset_path, 'test', 'images')
    test_labels_path = os.path.join(dataset_path, 'test', 'labels')

    os.makedirs(valid_images_path, exist_ok=True)
    os.makedirs(valid_labels_path, exist_ok=True)
    os.makedirs(test_images_path, exist_ok=True)
    os.makedirs(test_labels_path, exist_ok=True)

    # Получение всех файлов изображений и соответствующих меток
    images = [f for f in os.listdir(train_images_path)]
    labels = [f for f in os.listdir(train_labels_path)]
    print(len(images))

    # Убедимся, что количество изображений и меток совпадает
    images.sort()
    labels.sort()

    # Проверка на соответствие количества изображений и меток
    if len(images) != len(labels):
        print("Количество изображений и меток не совпадает.")
        exit()

    # Перемешивание данных
    data = list(zip(images, labels))
    random.shuffle(data)
    images, labels = zip(*data)
    print(images)
    print(labels)

    # Разделение данных
    num_images = len(images)
    num_test = int(num_images * test_percent)
    num_valid = int(num_images * valid_percent)
    num_train = num_images - num_test - num_valid

    # Перемещение данных в соответствующие папки
    def move_files(file_list, source_image_dir, source_label_dir, dest_image_dir, dest_label_dir):
        for file in file_list:
            image_path = os.path.join(source_image_dir, file)
            label_path = os.path.join(source_label_dir, os.path.splitext(file)[0] + '.txt')
            shutil.move(image_path, os.path.join(dest_image_dir, file))
            shutil.move(label_path, os.path.join(dest_label_dir, os.path.splitext(file)[0] + '.txt'))

    # Перемещение тестовых данных
    move_files(images[:num_test], train_images_path, train_labels_path, test_images_path, test_labels_path)

    # Перемещение валидационных данных
    move_files(images[num_test:num_test + num_valid], train_images_path, train_labels_path, valid_images_path, valid_labels_path)

    # Оставшиеся данные остаются в папке train

    print(f"Перемещено {num_test} изображений в папку test.")
    print(f"Перемещено {num_valid} изображений в папку valid.")
    print(f"Осталось {num_train} изображений в папке train.")

    data_path = os.path.join(settings.BASE_DIR, "src", "data.yaml")
    settings.MODEL.train(data=data_path, epochs=5)
    settings.MODEL.save(os.path.join(settings.BASE_DIR, "model", "best.pt"))

    for filename in os.listdir(input_folder):
        file_path = os.path.join(input_folder, filename)
        if os.path.isfile(file_path):
            os.remove(file_path)

    dff = pd.DataFrame(columns=['main_class', 'filename', 'x_left_bottom', 'y_left_bottom', 'length', 'width'])
    dff.to_csv(os.path.join(settings.BASE_DIR, "src", "data.csv"), index=False)
