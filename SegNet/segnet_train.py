import argparse
import os
from PIL import Image
import numpy as np
import random
import matplotlib
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
import tensorflow.keras as keras
from tensorflow.keras.layers import Conv2D, BatchNormalization, MaxPooling2D, UpSampling2D, Reshape, Permute, Activation
import sys
sys.path.append("..")
from CONFIG import *

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
matplotlib.use("Agg")
classes = [0., 1., 2., 3., 4., 5., 6., 255.]
labelencoder = LabelEncoder()
labelencoder.fit(classes)
src_path = TRAIN_AUG_SRC_PATH
lab_path = TRAIN_AUG_LAB_PATH


def SegNet():
    model = keras.Sequential()
    # Encoder
    model.add(Conv2D(64, (3, 3), strides=(1, 1), input_shape=(IMG_W, IMG_H, 3), padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(64, (3, 3), strides=(1, 1), padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    # (128,128)
    model.add(Conv2D(128, (3, 3), strides=(1, 1), padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(128, (3, 3), strides=(1, 1), padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    # (64,64)
    model.add(Conv2D(256, (3, 3), strides=(1, 1), padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(256, (3, 3), strides=(1, 1), padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(256, (3, 3), strides=(1, 1), padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    # (32,32)
    model.add(Conv2D(512, (3, 3), strides=(1, 1), padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(512, (3, 3), strides=(1, 1), padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(512, (3, 3), strides=(1, 1), padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    # (16,16)
    model.add(Conv2D(512, (3, 3), strides=(1, 1), padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(512, (3, 3), strides=(1, 1), padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(512, (3, 3), strides=(1, 1), padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    # (8,8), Decoder
    model.add(UpSampling2D(size=(2, 2)))
    # (16,16)
    model.add(Conv2D(512, (3, 3), strides=(1, 1), padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(512, (3, 3), strides=(1, 1), padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(512, (3, 3), strides=(1, 1), padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(UpSampling2D(size=(2, 2)))
    # (32,32)
    model.add(Conv2D(512, (3, 3), strides=(1, 1), padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(512, (3, 3), strides=(1, 1), padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(512, (3, 3), strides=(1, 1), padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(UpSampling2D(size=(2, 2)))
    # (64,64)
    model.add(Conv2D(256, (3, 3), strides=(1, 1), padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(256, (3, 3), strides=(1, 1), padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(256, (3, 3), strides=(1, 1), padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(UpSampling2D(size=(2, 2)))
    # (128,128)
    model.add(Conv2D(128, (3, 3), strides=(1, 1), padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(128, (3, 3), strides=(1, 1), padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(UpSampling2D(size=(2, 2)))
    # (256,256)
    model.add(Conv2D(64, (3, 3), strides=(1, 1), input_shape=(3, IMG_W, IMG_H), padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(64, (3, 3), strides=(1, 1), padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(N_LABEL, (1, 1), strides=(1, 1), padding='same'))
    model.add(Reshape((N_LABEL, IMG_W * IMG_H)))
    model.add(Permute((2, 1)))
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])
    model.summary()
    return model


def load_img(path, grayscale=False):
    img = Image.open(path)
    if not grayscale:
        img = np.array(img, dtype="float") / 255.0
    return img


def get_train_val(val_rate=VAL_RATE):
    train_url = []
    train_set = []
    val_set = []
    for pic in os.listdir(src_path):
        train_url.append(pic)
    random.shuffle(train_url)
    total_num = len(train_url)
    val_num = int(val_rate * total_num)
    for i in range(total_num):
        if i < val_num:
            val_set.append(train_url[i])
        else:
            train_set.append(train_url[i])
    return train_set, val_set


# data for training
def generateData(batch_size, data=[]):
    while True:
        train_data = []
        train_label = []
        batch = 0
        for i in (range(len(data))):
            url = data[i]
            batch += 1
            img = load_img(src_path + url)
            img = keras.preprocessing.image.img_to_array(img)
            train_data.append(img)
            label = load_img(lab_path + url[0:8]+'png', grayscale=True)
            label = keras.preprocessing.image.img_to_array(label).reshape((IMG_W * IMG_H,))
            train_label.append(label)
            if batch % batch_size == 0:
                train_data = np.array(train_data)
                train_label = np.array(train_label).flatten()
                train_label = labelencoder.transform(train_label)
                train_label = keras.utils.to_categorical(train_label, num_classes=N_LABEL)
                train_label = train_label.reshape((batch_size, IMG_H * IMG_W, N_LABEL))
                yield train_data, train_label
                train_data = []
                train_label = []
                batch = 0


# data for validation
def generateValidData(batch_size, data=[]):
    while True:
        valid_data = []
        valid_label = []
        batch = 0
        for i in (range(len(data))):
            url = data[i]
            batch += 1
            img = load_img(src_path + url)
            img = keras.preprocessing.image.img_to_array(img)
            valid_data.append(img)
            label = load_img(lab_path + url[0:8]+'png', grayscale=True)
            label = keras.preprocessing.image.img_to_array(label).reshape((IMG_W * IMG_H,))
            valid_label.append(label)
            if batch % batch_size == 0:
                valid_data = np.array(valid_data)
                valid_label = np.array(valid_label).flatten()
                valid_label = labelencoder.transform(valid_label)
                valid_label = keras.utils.to_categorical(valid_label, num_classes=N_LABEL)
                valid_label = valid_label.reshape((batch_size, IMG_H * IMG_W, N_LABEL))
                yield valid_data, valid_label
                valid_data = []
                valid_label = []
                batch = 0


# training
def train(args):
    model = SegNet()
    modelcheck = keras.callbacks.ModelCheckpoint(args['model'], monitor='val_accuracy', save_best_only=True, mode='max')
    callable = [modelcheck]
    train_set, val_set = get_train_val()
    train_num = len(train_set)
    valid_num = len(val_set)
    print("The number of train data is", train_num)
    print("The number of val data is", valid_num)
    H = model.fit_generator(generator=generateData(SEGNET_BATCHSIZE, train_set),
                            steps_per_epoch=train_num // SEGNET_BATCHSIZE,
                            epochs=SEGNET_EPOCHS, verbose=1,
                            validation_data=generateValidData(SEGNET_BATCHSIZE, val_set),
                            validation_steps=valid_num // SEGNET_BATCHSIZE, callbacks=callable, max_queue_size=1)
    # plot the training loss and accuracy
    plt.style.use("ggplot")
    plt.figure()
    plt.plot(np.arange(0, SEGNET_EPOCHS), H.history["loss"], label="train_loss")
    plt.plot(np.arange(0, SEGNET_EPOCHS), H.history["val_loss"], label="val_loss")
    plt.plot(np.arange(0, SEGNET_EPOCHS), H.history["accuracy"], label="train_accuracy")
    plt.plot(np.arange(0, SEGNET_EPOCHS), H.history["val_accuracy"], label="val_accuracy")
    plt.title("Training Loss and Accuracy on SegNet")
    plt.xlabel("Epoch #")
    plt.ylabel("Loss/Accuracy")
    plt.legend(loc="lower left")
    plt.savefig(args["plot"])


def args_parse():
    # construct the argument parse and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-a", "--augment", help="using data augment or not", action="store_true", default=False)
    ap.add_argument("-m", "--model", type=str, default="SgeNet.h5", help="path to output model")
    ap.add_argument("-p", "--plot", type=str, default="SegNet.png",
                    help="path to output accuracy/loss plot")
    args = vars(ap.parse_args())
    return args


if __name__ == '__main__':
    args = args_parse()
    if not args['augment']:
        src_path = TRAIN_SRC_PATH
        lab_path = TRAIN_LAB_PATH
    train(args)
