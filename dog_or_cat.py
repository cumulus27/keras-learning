#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
A CNN test.
"""

import os, shutil
from keras import layers
from keras import models
from keras import optimizers
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt

from keras.applications import VGG16


class DogCat(object):

    def __init__(self, data_path):
        self.data_path = data_path
        self.model = None
        self.history = None
        self.train_generator = None
        self.validation_generator = None
        self.conv_base = None
        self.model_file_name = ""
        self.train_dir = ""
        self.validation_dir = ""
        self.test_dir = ""

    def get_data_path(self):
        base_dir = self.data_path + '/cats_and_dogs_small'  # ←------ 保存较小数据集的目录

        train_dir = os.path.join(base_dir, 'train')  # （以下5行）分别对应划分后的训练、验证和测试的目录
        validation_dir = os.path.join(base_dir, 'validation')
        test_dir = os.path.join(base_dir, 'test')

        self.train_dir = train_dir
        self.validation_dir = validation_dir
        self.test_dir = test_dir

    def load_data(self):

        original_dataset_dir = self.data_path + '/train'  # ←------ 原始数据集解压目录的路径

        base_dir = self.data_path + '/cats_and_dogs_small'  # ←------ 保存较小数据集的目录

        os.mkdir(base_dir)

        train_dir = os.path.join(base_dir, 'train')          #（以下5行）分别对应划分后的训练、验证和测试的目录
        os.mkdir(train_dir)
        validation_dir = os.path.join(base_dir, 'validation')
        os.mkdir(validation_dir)
        test_dir = os.path.join(base_dir, 'test')
        os.mkdir(test_dir)

        train_cats_dir = os.path.join(train_dir, 'cats')     #（以下2行）猫的训练图像目录
        os.mkdir(train_cats_dir)

        train_dogs_dir = os.path.join(train_dir, 'dogs')    #（以下2行）狗的训练图像目录
        os.mkdir(train_dogs_dir)

        validation_cats_dir = os.path.join(validation_dir, 'cats')       #（以下2行）猫的验证图像目录
        os.mkdir(validation_cats_dir)

        validation_dogs_dir = os.path.join(validation_dir, 'dogs')    #（以下2行）狗的测试图像目录
        os.mkdir(validation_dogs_dir)

        test_cats_dir = os.path.join(test_dir, 'cats')    #（以下2行）猫的测试图像目录
        os.mkdir(test_cats_dir)

        test_dogs_dir = os.path.join(test_dir, 'dogs')    #（以下2行）狗的测试图像目录
        os.mkdir(test_dogs_dir)

        fnames = ['cat.{}.jpg'.format(i) for i in range(1000)]   #（以下5行）将前1000张猫的图像复制到train_cats_dir
        for fname in fnames:
            src = os.path.join(original_dataset_dir, fname)
            dst = os.path.join(train_cats_dir, fname)
            shutil.copyfile(src, dst)

        fnames = ['cat.{}.jpg'.format(i) for i in range(1000, 1500)]   #（以下5行）将接下来500张猫的图像复制到validation_cats_dir
        for fname in fnames:
            src = os.path.join(original_dataset_dir, fname)
            dst = os.path.join(validation_cats_dir, fname)
            shutil.copyfile(src, dst)

        fnames = ['cat.{}.jpg'.format(i) for i in range(1500, 2000)]  #（以下5行）将接下来的500张猫的图像复制到test_cats_dir
        for fname in fnames:
            src = os.path.join(original_dataset_dir, fname)
            dst = os.path.join(test_cats_dir, fname)
            shutil.copyfile(src, dst)

        fnames = ['dog.{}.jpg'.format(i) for i in range(1000)]      #（以下5行）将前1000张狗的图像复制到train_dogs_dir
        for fname in fnames:
            src = os.path.join(original_dataset_dir, fname)
            dst = os.path.join(train_dogs_dir, fname)
            shutil.copyfile(src, dst)

        fnames = ['dog.{}.jpg'.format(i) for i in range(1000, 1500)]      #（以下5行）将接下来500张狗的图像复制到validation_dogs_dir
        for fname in fnames:
            src = os.path.join(original_dataset_dir, fname)
            dst = os.path.join(validation_dogs_dir, fname)
            shutil.copyfile(src, dst)

        fnames = ['dog.{}.jpg'.format(i) for i in range(1500, 2000)]    #（以下5行）将接下来500张狗的图像复制到test_dogs_dir
        for fname in fnames:
            src = os.path.join(original_dataset_dir, fname)
            dst = os.path.join(test_dogs_dir, fname)
            shutil.copyfile(src, dst)

    def generate_simple_model(self):
        model = models.Sequential()
        model.add(layers.Conv2D(32, (3, 3), activation='relu',
                                input_shape=(150, 150, 3)))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Conv2D(64, (3, 3), activation='relu'))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Conv2D(128, (3, 3), activation='relu'))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Conv2D(128, (3, 3), activation='relu'))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Flatten())
        model.add(layers.Dense(512, activation='relu'))
        model.add(layers.Dense(1, activation='sigmoid'))

        self.model = model
        self.model_file_name = 'cats_and_dogs_small_1.h5'

    def generate_model_with_dropout(self):
        model = models.Sequential()
        model.add(layers.Conv2D(32, (3, 3), activation='relu',
                                input_shape=(150, 150, 3)))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Conv2D(64, (3, 3), activation='relu'))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Conv2D(128, (3, 3), activation='relu'))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Conv2D(128, (3, 3), activation='relu'))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Flatten())
        model.add(layers.Dropout(0.5))
        model.add(layers.Dense(512, activation='relu'))
        model.add(layers.Dense(1, activation='sigmoid'))

        self.model = model
        self.model_file_name = 'cats_and_dogs_small_2.h5'

    def load_VGG16_as_base_conv(self):
        self.conv_base = VGG16(weights='imagenet',
                          include_top=False,
                          input_shape=(150, 150, 3))

    def generate_dense_model_on_base_conv(self):
        model = models.Sequential()
        model.add(self.conv_base)
        model.add(layers.Flatten())
        model.add(layers.Dense(256, activation='relu'))
        model.add(layers.Dense(1, activation='sigmoid'))

        self.model = model
        self.model_file_name = 'cats_and_dogs_small_3.h5'

    def freeze_all_base_conv(self):
        self.conv_base.trainable = False

    def freeze_most_conv(self):
        self.conv_base.trainable = True

        set_trainable = False
        for layer in self.conv_base.layers:
            if layer.name == 'block5_conv1':
                set_trainable = True
            if set_trainable:
                layer.trainable = True
            else:
                layer.trainable = False

    def generate_simple_generator(self):
        train_datagen = ImageDataGenerator(rescale=1. / 255)
        test_datagen = ImageDataGenerator(rescale=1. / 255)

        self.train_generator = train_datagen.flow_from_directory(
            self.train_dir,
            target_size=(150, 150),
            batch_size=20,
            class_mode='binary')

        self.validation_generator = test_datagen.flow_from_directory(
            self.validation_dir,
            target_size=(150, 150),
            batch_size=20,
            class_mode='binary')

    def generate_enhance_generator(self):
        train_datagen = ImageDataGenerator(
            rescale=1. / 255,
            rotation_range=40,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            fill_mode='nearest')

        test_datagen = ImageDataGenerator(rescale=1. / 255)

        self.train_generator = train_datagen.flow_from_directory(
            self.train_dir,
            target_size=(150, 150),
            batch_size=32,
            class_mode='binary')

        self.validation_generator = test_datagen.flow_from_directory(
            self.validation_dir,
            target_size=(150, 150),
            batch_size=32,
            class_mode='binary')

    def compile_model(self, lr=1e-4):
        self.model.compile(loss='binary_crossentropy',
                      optimizer=optimizers.RMSprop(lr=lr),
                      metrics=['acc'])

    def fit_model(self, steps_per_epoch=100, epochs=30, validation_steps=50):
        self.history = self.model.fit_generator(
            self.train_generator,
            steps_per_epoch=steps_per_epoch,
            epochs=epochs,
            validation_data=self.validation_generator,
            validation_steps=validation_steps)

        self.model.save(os.path.join(self.data_path, self.model_file_name))

    def plot_history(self):
        acc = self.history.history['acc']
        val_acc = self.history.history['val_acc']
        loss = self.history.history['loss']
        val_loss = self.history.history['val_loss']

        epochs = range(1, len(acc) + 1)

        plt.plot(epochs, acc, 'bo', label='Training acc')
        plt.plot(epochs, val_acc, 'b', label='Validation acc')
        plt.title('Training and validation accuracy')
        plt.legend()

        plt.figure()

        plt.plot(epochs, loss, 'bo', label='Training loss')
        plt.plot(epochs, val_loss, 'b', label='Validation loss')
        plt.title('Training and validation loss')
        plt.legend()

        plt.show()


if __name__ == "__main__":
    path = "/home/py/data/dogs-vs-cats"

    cnn = DogCat(path)
    cnn.load_data()
    cnn.get_data_path()

    # cnn.generate_simple_model()
    # cnn.generate_model_with_dropout()
    # cnn.model.summary()

    cnn.load_VGG16_as_base_conv()
    cnn.generate_dense_model_on_base_conv()
    cnn.freeze_all_base_conv()
    # cnn.conv_base.summary()

    # cnn.generate_simple_generator()
    cnn.generate_enhance_generator()
    cnn.compile_model(lr=1e-5)
    cnn.fit_model(epochs=30)

    cnn.plot_history()





