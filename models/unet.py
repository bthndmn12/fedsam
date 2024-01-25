import os
import keras
from models.lossfunc.diceloss import DiceBCELoss
from tqdm import tqdm
from glob import glob
import tensorflow as tf
from numpy import zeros
from numpy.random import randint
from tensorflow.image import resize
from keras.preprocessing.image import load_img, img_to_array
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import pandas as pds
from tensorflow.keras.layers import Conv2D, BatchNormalization, Activation, MaxPool2D, Conv2DTranspose, Concatenate, Input, multiply, concatenate, add, Layer, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.utils import plot_model
from tensorflow.keras.callbacks import Callback, ModelCheckpoint

class ShowProgress(Callback):
    def on_epoch_end(self, epoch, logs=None):
        print(f"Epoch {epoch+1} completed")

class UNetModel:
    def __init__(self):
        self.UNet = self.build_model()

    def build_model(self):
        class Encoder(Layer):
            def __init__(self, filters, rate, pooling=True, **kwargs):
                super(Encoder, self).__init__(**kwargs)
                self.filters = filters
                self.rate = rate
                self.pooling = pooling
                self.bn = BatchNormalization()
                self.c1 = Conv2D(filters, kernel_size=3, padding='same', activation='relu', kernel_initializer="he_normal")
                self.drop = Dropout(rate)
                self.c2 = Conv2D(filters, kernel_size=3, padding='same', activation='relu', kernel_initializer="he_normal")
                self.drop2 = Dropout(rate)
                self.c3 = Conv2D(filters, kernel_size=3, padding='same', activation='relu', kernel_initializer="he_normal")
                self.pool = MaxPool2D()

            def call(self, X):
                x = self.bn(X)
                x = self.c1(x)
                x = self.drop(x)
                x = self.c2(x)
                x = self.drop2(x)
                x = self.c3(x)

                if self.pooling:
                    y = self.pool(x)
                    return y, x
                else:
                    return x

            def get_config(self):
                base_config = super().get_config()
                return {
                    **base_config,
                    "filters": self.filters,
                    "rate": self.rate,
                    "pooling": self.pooling
                }
            def get_parameters(self, config):
                return self.UNet.get_weights()
            
            def set_parameters(self, weights):
                """Set model parameters from a list of NumPy ndarrays."""
                self.UNet.set_weights(weights)

        class Decoder(Layer):
            def __init__(self, filters, rate, **kwargs):
                super(Decoder, self).__init__(**kwargs)
                self.filters = filters
                self.rate = rate
                self.bn = BatchNormalization()
                self.cT = Conv2DTranspose(filters, kernel_size=3, strides=2, padding='same', activation='relu', kernel_initializer="he_normal")
                self.net = Encoder(filters, rate, pooling=False)

            def call(self, X):
                x, skip_x = X
                x = self.bn(x)
                x = self.cT(x)
                x = concatenate([x, skip_x])
                x = self.net(x)
                return x

            def get_config(self):
                base_config = super().get_config()
                return {
                    **base_config,
                    "filters": self.filters,
                    "rate": self.rate,
                }

        # Inputs
        unet_inputs = Input(shape=(256, 256, 3), name="UNetInput")

        # Encoder Network : Downsampling phase
        p1, c1 = Encoder(64, 0.1, name="Encoder1")(unet_inputs)
        p2, c2 = Encoder(128, 0.1, name="Encoder2")(p1)
        p3, c3 = Encoder(256, 0.2, name="Encoder3")(p2)
        p4, c4 = Encoder(512, 0.2, name="Encoder4")(p3)

        # Encoding Layer : Latent Representation
        e = Encoder(512, 0.3, pooling=False)(p4)

        # Attention + Decoder Network : Upsampling phase.
        d1 = Decoder(512, 0.2, name="Decoder1")([e, c4])
        d2 = Decoder(256, 0.2, name="Decoder2")([d1, c3])
        d3 = Decoder(128, 0.1, name="Decoder3")([d2, c2])
        d4 = Decoder(64, 0.1, name="Decoder4")([d3, c1])

        # Output
        unet_out = Conv2D(3, kernel_size=3, padding='same', activation='sigmoid')(d4)

        # Model
        model = Model(
            inputs=unet_inputs,
            outputs=unet_out,
            name="AttentionUNet"
        )

        # Compiling
        model.compile(
            loss=DiceBCELoss,
            optimizer='adam'
        )

        return model
    
    
