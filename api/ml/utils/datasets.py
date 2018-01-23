import os
import numpy as np
from scipy.io import loadmat
from random import shuffle
import pandas as pd
from .image_processing import resize_img_with_cv2, gray2bgr


dataset_dir = os.path.join(os.path.pardir, 'ml_dataset')
fer2013 = {
    'dataset_name': 'fer2013',
    'input_size': (64, 64),
    'labels': ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral'],
    'data_file': 'fer2013.csv'
}
datasets = [fer2013]

def exist_dataset(dataset_name):
    for i, dataset in enumerate(datasets):
        if dataset['dataset_name'] == dataset_name:
            return True
    return False

def get_dataset_info(dataset_name):
    for i, dataset in enumerate(datasets):
        if dataset['dataset_name'] == dataset_name:
            return dataset
        else:
            raise ValueError('해당 데이터셋이 존재하지 않습니다.')

def add_classification_dataset(dataset_name, input_size, labels):
    if exist_dataset(dataset_name):
        print('\'%s\' 데이터셋이 이미 존재합니다.' % (dataset_name))
        return
    dataset = {}
    dataset['dataset_name'] = dataset_name
    dataset['input_size'] = input_size
    dataset['labels'] = labels
    datasets.append(dataset)
    print('\'%s\' 데이터셋이 추가 되었습니다.' % (dataset_name))

def remove_dataset(dataset_name):
    for i, dataset in enumerate(datasets):
        if dataset['dataset_name'] == dataset_name:
            datasets.pop(i)

def get_dataset_list():
    names = []
    paths = []

    dir_list = os.listdir(dataset_dir)
    for i, dataset in enumerate(dir_list):
        path = os.path.join(dataset_dir, dataset)
        if os.path.isdir(path):
            names.append(dataset)
            paths.append(path)

    return names, paths

class DatasetManager:
    def __init__(self, dataset_name='fer2013', input_size=None):
        dataset_info = get_dataset_info(dataset_name)
        self.dataset_name = dataset_info['dataset_name']
        self.dataset_path = os.path.join(dataset_dir, dataset_name, dataset_info['data_file'])
        if input_size is None:
            self.input_size = dataset_info['input_size']
        else:
            self.input_size = input_size

    def get_data(self):
        ground_truth_data = None
        if self.dataset_name == 'fer2013':
            ground_truth_data = self._load_fer2013()
        else:
            pass

        return ground_truth_data

    def _load_fer2013(self):
        data = pd.read_csv(self.dataset_path)
        pixels = data['pixels'].tolist()
        width, height = 48, 48
        faces = []
        for pixel_sequence in pixels:
            face = [int(pixel) for pixel in pixel_sequence.split(' ')]
            face = np.asarray(face).reshape(width, height)
            # face = cv2.resize(face.astype('uint8'), self.input_size)
            channel = self.input_size[:3]
            if channel == 3:
                face = gray2bgr(face.astype('uint8'))
            face = resize_img_with_cv2(face, self.input_size)
            faces.append(face.astype('float32'))
        faces = np.asarray(faces)
        faces = np.expand_dims(faces, -1)
        emotions = pd.get_dummies(data['emotion']).as_matrix()
        return faces, emotions

def split_data(x, y, validation_split=.2):
    num_samples = len(x)
    num_train_samples = int((1 - validation_split)*num_samples)
    train_x = x[:num_train_samples]
    train_y = y[:num_train_samples]
    val_x = x[num_train_samples:]
    val_y = y[num_train_samples:]
    train_data = (train_x, train_y)
    val_data = (val_x, val_y)
    return train_data, val_data


def get_labels(dataset_name):
    dataset_info = get_dataset_info(dataset_name)
    labels = {}
    for i, label in enumerate(dataset_info['labels']):
        labels[i] = label

    return labels

def get_class_to_arg(dataset_name):
    dataset_info = get_dataset_info(dataset_name)
    classes = {}
    for i, label in enumerate(dataset_info['labels']):
        classes[label] = i

    return classes

def divide_dataset(x_data, y_data, train_rate=0.6, valid_rate=0.2, test_rate=0.2):
    if train_rate + valid_rate + test_rate != 1:
        print("train_rate, valid_rate, test_rate 세 값의 총합이 반드시 1이 되어야 합니다.")
        return
    total_cnt = len(y_data)
    randidx = np.random.randint(total_cnt, size=total_cnt)
    trainidx = randidx[0:int(total_cnt * train_rate)]
    valididx = randidx[int(total_cnt * train_rate):int(total_cnt * (train_rate+valid_rate))]
    testidx = randidx[int(total_cnt * (train_rate+valid_rate)):total_cnt]
    x_train = x_data[trainidx, :]
    y_train = y_data[trainidx, :]
    x_valid = x_data[valididx, :]
    y_valid = y_data[valididx, :]
    x_test = x_data[testidx, :]
    y_test = y_data[testidx, :]

    return x_train, y_train, x_valid, y_valid, x_test, y_test

def get_nth_batch(x_data, y_data, batch_idx=0, batch_size=64):
    total_cnt = len(y_data)
    if batch_size > total_cnt:
        print('배치 사이즈가 전체 사이즈보다 큽니다. (Total: %d, Batch size: %d)' % (total_cnt, batch_size))
        return
    full_batch, remained = divmod(total_cnt, batch_size)
    total_batch = full_batch
    if remained > 0:
        total_batch += 1

    start_idx = batch_idx * batch_size
    end_idx = (batch_idx+1) * batch_size
    if batch_idx + 1 > total_batch:
        end_idx = total_cnt
    batch_xs = x_data[start_idx:end_idx, :]
    batch_ys = y_data[start_idx:end_idx, :]

    return batch_xs, batch_ys


# randidx    = np.random.randint(imgcnt, size=imgcnt)
# trainidx   = randidx[0:int(3*imgcnt/5)]
# testidx    = randidx[int(3*imgcnt/5):imgcnt]
# trainimg   = totalimg[trainidx, :]
# trainlabel = totallabel[trainidx, :]
# testimg    = totalimg[testidx, :]
# testlabel  = totallabel[testidx, :]