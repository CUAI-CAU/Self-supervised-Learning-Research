import os
import pandas as pd
import numpy as np
import shutil
np.random.seed(10)

path = 'data/datasets/imagenet_200'

def save_file(class_path, save_path, filenames, start_num, end_num):
    target_filenames = filenames[start_num:end_num]
    for target_filename in target_filenames:
        src = os.path.join(class_path, target_filename)
        dst = os.path.join(save_path, target_filename)
        shutil.copy(src, dst)

def split_train_test(base_path):
    for class_nm in os.listdir(path):
        class_path = os.path.join(path, class_nm)
        filenames = os.listdir(class_path)

        train_path = os.path.join('dataset/', 'Training', class_nm)
        test_path = os.path.join('dataset/', 'test', class_nm)

        if not os.path.exists(train_path):
            os.makedirs(train_path)
        if not os.path.exists(test_path):
            os.makedirs(test_path)

        if len(filenames) > 70 :
            save_file(class_path, train_path, filenames, 0, 70)
            save_file(class_path, test_path, filenames, 70, 100)
        else :
            print('>> check class name : ', class_nm )
            print('>> data lenghts : ', len(filenames))
            save_file(class_path, train_path, filenames, 0, int(len(filenames) * 0.7))
            save_file(class_path, test_path, filenames, int(len(filenames) * 0.7), len(filenames))


if __name__ == '__main__' :
    split_train_test(path)