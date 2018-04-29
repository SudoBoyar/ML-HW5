import os

import numpy as np
from scipy.misc import imread
from skimage.color import rgb2lab
from sklearn.model_selection import train_test_split


def load_data(folder):
    training_on = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i']
    class_repr = {training_on[i]: [0] * i + [1] + [0] * (len(training_on) - i - 1) for i in range(len(training_on))}

    src_paths = []
    classes = []
    for subject in os.listdir(folder):
        subject_folder = os.path.join(folder, subject)
        for class_folder in os.listdir(subject_folder):
            if class_folder not in training_on:
                continue
            class_path = os.path.join(subject_folder, class_folder)
            for i, f in enumerate(sorted(os.listdir(class_path))):
                if f[:5] != 'color':
                    break
                if i % 3 == 0:
                    src_paths.append(os.path.join(class_path, f))
                    classes.append(class_repr[class_folder])

    images = np.zeros((len(src_paths), 64, 64, 3))
    classes = np.array(classes)
    for i, sample_path in enumerate(src_paths):
        img = imread(sample_path)
        shape = (64, 64, 3)
        img.resize(shape)
        images[i] = rgb2lab(img)

    return train_test_split(images, classes, test_size=.2)
