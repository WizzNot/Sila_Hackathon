import cv2
import os
import numpy as np
import pandas as pd
from django.conf import settings

df = pd.read_csv(settings.BASE_DIR + "/src/data.csv", delimiter=",")

input_folder = settings.BASE_DIR + '/src/learn_images'
output_folder = settings.BASE_DIR + '/src/dataset/train'
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
            f.write(f"{dict_classes[df[df['filename']==filename].values[0][0]]} {x_center} {y_center} {width} {height}\n")
        print(f"Processed and saved {filename}")

# SECOND PART

import os
import shutil
import random

# Параметры для разделения данных
test_percent = 0.2  # Процент данных для тестирования
valid_percent = 0.1  # Процент данных для проверки

# Путь к папке с данными
dataset_path = settings.BASE_DIR + '/src/dataset'
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
settings.MODEL.train(data=data_path, epochs=300)

for filename in os.listdir(input_folder):
    file_path = os.path.join(input_folder, filename)
    if os.path.isfile(file_path):
        os.remove(file_path)

dff = pd.DataFrame(columns=['main_class', 'filename', 'x_left_bottom', 'y_left_bottom', 'length', 'width'])
dff.to_csv(settings.BASE_DIR + "src/data.csv", index=False)
