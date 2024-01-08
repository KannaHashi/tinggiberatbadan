import os
import shutil
from sklearn.model_selection import train_test_split

def copy_data(files, input_path, output_path):
    os.makedirs(output_path, exist_ok=True)
    for file in files:
        shutil.copy(os.path.join(input_path, file), os.path.join(output_path, file))

def create_annotations(image_files, input_path, output_path):
    os.makedirs(output_path, exist_ok=True)
    for file in image_files:
        image_name, _ = os.path.splitext(file)
        label_file_path = os.path.join(input_path, image_name + '.txt')
        output_file_path = os.path.join(output_path, image_name + '.txt')

        with open(label_file_path, 'r') as label_file, open(output_file_path, 'w') as output_file:
            output_file.write(label_file.read())

# Tentukan path dataset YOLOv5 PyTorch
dataset_path = 'personDataSet'
output_path = 'dataset_output'
os.makedirs(output_path, exist_ok=True)

data_train_path = os.path.join(dataset_path, 'train')
data_test_path = os.path.join(dataset_path, 'test')

# Memisahkan dataset menjadi train dan test
image_files_train = [f for f in os.listdir(os.path.join(data_train_path, 'images')) if f.endswith(('.jpg', '.png', '.jpeg'))]
image_files_test = [f for f in os.listdir(os.path.join(data_test_path, 'images')) if f.endswith(('.jpg', '.png', '.jpeg'))]

label_files_train = [f for f in os.listdir(os.path.join(data_train_path, 'labels')) if f.endswith(('.txt'))]
label_files_test = [f for f in os.listdir( os.path.join(data_test_path, 'labels')) if f.endswith(('.txt'))]

# Menyalin gambar ke direktori train
train_output_path = os.path.join(output_path, 'train', 'images')
copy_data(image_files_train, os.path.join(data_train_path, 'images'), train_output_path)

# Menyalin gambar ke direktori test
test_output_path = os.path.join(output_path, 'test', 'images')
copy_data(image_files_test, os.path.join(data_test_path, 'images'), test_output_path)

# Menyalin file anotasi ke direktori train
train_annotation_output_path = os.path.join(output_path, 'train', 'labels')
copy_data(label_files_train, os.path.join(data_train_path, 'labels'), train_annotation_output_path)

# Menyalin file anotasi ke direktori test
test_annotation_output_path = os.path.join(output_path, 'test', 'labels')
copy_data(label_files_test, os.path.join(data_test_path, 'labels'), test_annotation_output_path)
