# -*- coding: utf-8 -*-

import numpy as np
from keras.layers import (Input, Dense, Reshape, BatchNormalization, Conv2D, MaxPooling2D, Flatten)
from keras import Model
from keras.optimizers import RMSprop
from keras.layers.advanced_activations import LeakyReLU
from keras import backend as K
import keras
import tensorflow as tf


def wasserstein_loss(y_true, y_pred):
    half_sample_size  = tf.cast(tf.shape(y_pred)[0]/2, dtype=tf.int32)
    y_pred_r = y_pred[:half_sample_size]
    y_pred_i = y_pred[half_sample_size:]
    return -K.mean(y_pred_r - y_pred_i - 2)


class WassersteinGAN:
    """Wasserstein GAN model optimzied on EM"""
    def __init__(self, num_row, num_col, num_channel):
        self.num_row = num_row
        self.num_col = num_col
        self.num_channel = num_channel
        self.input_shape_g = 100
        self.num_critic = 5
        self.clip_value = 0.01
        self.generator = self.discriminator = None 
        self.opt = RMSprop(lr=0.00005)
        self.images=[]


    def build_generator(self, summary=False):
        """Generator networks here"""
        inputs = Input(shape=(self.input_shape_g,))
        x = Dense((self.num_row+9)*(self.num_col+9)*self.num_channel, activation='relu')(inputs)
        x = Reshape((self.num_row+9, self.num_col+9, self.num_channel))(x)
        x = BatchNormalization(momentum=0.8)(x)
        x = Conv2D(32, kernel_size=4, activation='relu')(x)
        x = BatchNormalization(momentum=0.8)(x)
        x = Conv2D(16, kernel_size=4, activation='relu')(x)
        x = BatchNormalization(momentum=0.8)(x)
        outputs = Conv2D(self.num_channel, kernel_size=4, activation='sigmoid')(x)
        generator = Model(inputs=inputs, outputs=outputs)

        if summary:
            generator.summary()
        return generator


    def build_discriminator(self, summary=False):
        """Discriminator networks here"""
        inputs = Input(shape=(self.num_row, self.num_col, self.num_channel))
        x = Conv2D(16, kernel_size=3)(inputs)
        x = LeakyReLU(alpha=0.2)(x)
        # x = MaxPooling2D(pool_size=(3,3))(x)
        x = Conv2D(32, kernel_size=3)(x)
        x = LeakyReLU(alpha=0.2)(x)
        # x = MaxPooling2D(pool_size=(3,3))(x)
        x = Conv2D(64, kernel_size=3)(x)
        x = LeakyReLU(alpha=0.2)(x)
        # x = MaxPooling2D(pool_size=(3,3))(x)
        x = Flatten()(x)
        outputs = Dense(1, activation='tanh')(x)
        discriminator = Model(inputs=inputs, outputs=outputs)

        if summary:
            discriminator.summary()
        return discriminator


    def build_gan(self, summary=False):
        self.generator = self.build_generator(summary=summary)
        self.generator.compile(loss='mse', optimizer=self.opt)
        self.discriminator = self.build_discriminator(summary=summary)
        self.discriminator.compile(loss=wasserstein_loss, optimizer=self.opt,
                                   metrics=[wasserstein_loss])
        z = Input(shape=(self.input_shape_g,))
        img = self.generator(z)
        discriminator_combine = self.build_discriminator(summary=summary)
        discriminator_combine.trainable = False
        validation = discriminator_combine(img)
        self.combined = Model(z, validation)
        self.combined.compile(loss='mse', optimizer=self.opt,
                              metrics=['mse'])


    def fit(self, X_train, y_train=None, iter_max=100, epochs=10000, batch_size=128, save_interval=10):
        half_batch_szie = int(batch_size / 2)
        assert half_batch_szie == batch_size / 2
        early = keras.callbacks.EarlyStopping(
            monitor='loss',
            min_delta=0,
            patience=10,
            verbose=0,
            mode='auto'
        )
        for num_iter in range(iter_max):
            print(num_iter)
            for _ in range(self.num_critic):
                idx = np.random.randint(0, X_train.shape[0], half_batch_szie)
                imgs = X_train[idx]
                noise = np.random.normal(0, 1, (half_batch_szie, self.input_shape_g))
                gen_imgs = self.generator.predict(noise)
                train_imgs = np.vstack([imgs, gen_imgs])
                train_targets = np.vstack([np.ones((half_batch_szie, 1)),
                                           -np.ones((half_batch_szie, 1))])
                self.discriminator.fit(x=train_imgs, y=train_targets, batch_size=batch_size,
                                       shuffle=False, epochs=epochs, callbacks=[early], verbose=0)
                weights =  self.discriminator.get_weights()
                num_layer_discriminator = len(weights)
                for i in range(num_layer_discriminator):
                    weights[i] = np.clip(weights[i], -self.clip_value, self.clip_value)
                self.discriminator.set_weights(weights)

            weights_combined = self.combined.get_weights()
            weights_combined[-num_layer_discriminator:] = weights
            self.combined.set_weights(weights_combined)

            noise = np.random.normal(0, 1, (batch_size, self.input_shape_g))
            self.combined.fit(x=noise, y=np.ones((batch_size, 1)), batch_size=batch_size,
                              epochs=epochs, callbacks=[early], verbose=0)
            if num_iter%save_interval == 0:
                noise = np.random.normal(0, 1, (2, self.input_shape_g))
                self.images.append(self.generator.predict(noise))





