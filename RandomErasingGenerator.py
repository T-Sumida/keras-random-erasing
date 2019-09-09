# coding:utf-8

from keras.preprocessing.image import ImageDataGenerator
from keras.utils import Sequence
import numpy as np


class RandomErasingGenerator(Sequence):
    def __init__(self, dirpath, target_size, batch_size,
                 p=0.5, sl=0.02, sh=0.4, r1=0.3, r2=3.3,
                 featurewise_center=False, samplewise_center=False,
                 featurewise_std_normalization=False,
                 samplewise_std_normalization=False,
                 zca_whitening=False, zca_epsilon=1e-06, rotation_range=0.0,
                 width_shift_range=0.0, height_shift_range=0.0,
                 brightness_range=None, shear_range=0.0, zoom_range=0.0,
                 channel_shift_range=0.0, fill_mode='nearest', cval=0.0,
                 horizontal_flip=False, vertical_flip=False, rescale=None,
                 preprocessing_function=None, data_format=None,
                 validation_split=0.0):
        image_gen = ImageDataGenerator(
            featurewise_center, samplewise_center,
            featurewise_std_normalization,
            samplewise_std_normalization,
            zca_whitening, zca_epsilon,
            rotation_range, width_shift_range,
            height_shift_range, brightness_range,
            shear_range, zoom_range,
            channel_shift_range, fill_mode, cval,
            horizontal_flip, vertical_flip,
            rescale, preprocessing_function,
            data_format, validation_split)
        self.gen = image_gen.flow_from_directory(
            dirpath, target_size=target_size,
            batch_size=batch_size, class_mode='categorical')
        self.p = p
        self.sl = sl
        self.sh = sh
        self.r1 = r1
        self.r2 = r2

    def __getitem__(self, idx):
        X, y = next(self.gen)
        batch_X = []
        for x in X:
            batch_X.append(self.random_erasing(x))
        return np.array(batch_X), y

    def __len__(self):
        return self.gen.__len__()

    def random_erasing(self, img):
        target = img.copy()
        if self.p < np.random.rand():
            return target

        h, w, _ = target.shape
        s = h*w

        while True:
            Se = np.random.uniform(self.sl, self.sh) * s
            re = np.random.uniform(self.r1, self.r2)

            He = int(np.sqrt(Se * re))
            We = int(np.sqrt(Se / re))

            xe = np.random.randint(0, w)
            ye = np.random.randint(0, h)

            if xe + We <= w and ye + He <= h:
                break
        mask = np.random.randint(0, 255, (He, We, 3))  # 矩形がを生成 矩形内の値はランダム値
        target[ye:ye + He, xe:xe + We, :] = mask  # 画像に矩形を重畳

        return target
