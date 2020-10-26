import argparse
import tensorflow.keras as keras
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, concatenate
from sklearn.preprocessing import LabelEncoder
import sys
import os
import matplotlib
import numpy as np
import random
from PIL import Image
import matplotlib.pyplot as plt
sys.path.append("..")
from CONFIG import *


os.environ["CUDA_VISIBLE_DEVICES"] = "0"
matplotlib.use("Agg")
classes = [0., 1., 2., 3., 4., 5., 6., 255.]
labelencoder = LabelEncoder()
labelencoder.fit(classes)
src_path = TRAIN_AUG_SRC_PATH
lab_path = TRAIN_AUG_LAB_PATH


def UNet():
    inputs = Input((IMG_W, IMG_H, 3))

    conv1 = Conv2D(32, (3, 3), activation="relu", padding="same")(inputs)
    conv1 = Conv2D(32, (3, 3), activation="relu", padding="same")(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Conv2D(64, (3, 3), activation="relu", padding="same")(pool1)
    conv2 = Conv2D(64, (3, 3), activation="relu", padding="same")(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Conv2D(128, (3, 3), activation="relu", padding="same")(pool2)
    conv3 = Conv2D(128, (3, 3), activation="relu", padding="same")(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = Conv2D(256, (3, 3), activation="relu", padding="same")(pool3)
    conv4 = Conv2D(256, (3, 3), activation="relu", padding="same")(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    conv5 = Conv2D(512, (3, 3), activation="relu", padding="same")(pool4)
    conv5 = Conv2D(512, (3, 3), activation="relu", padding="same")(conv5)

    up6 = concatenate([UpSampling2D(size=(2, 2))(conv5), conv4], axis=1)
    conv6 = Conv2D(256, (3, 3), activation="relu", padding="same")(up6)
    conv6 = Conv2D(256, (3, 3), activation="relu", padding="same")(conv6)

    up7 = concatenate([UpSampling2D(size=(2, 2))(conv6), conv3], axis=1)
    conv7 = Conv2D(128, (3, 3), activation="relu", padding="same")(up7)
    conv7 = Conv2D(128, (3, 3), activation="relu", padding="same")(conv7)

    up8 = concatenate([UpSampling2D(size=(2, 2))(conv7), conv2], axis=1)
    conv8 = Conv2D(64, (3, 3), activation="relu", padding="same")(up8)
    conv8 = Conv2D(64, (3, 3), activation="relu", padding="same")(conv8)

    up9 = concatenate([UpSampling2D(size=(2, 2))(conv8), conv1], axis=1)
    conv9 = Conv2D(32, (3, 3), activation="relu", padding="same")(up9)
    conv9 = Conv2D(32, (3, 3), activation="relu", padding="same")(conv9)

    conv10 = Conv2D(N_LABEL, (1, 1), activation="softmax")(conv9)

    model = keras.models.Model(inputs=inputs, outputs=conv10)
    model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['accuracy'])
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


def train(args):
    model = UNet()
    modelcheck = keras.callbacks.ModelCheckpoint(args['model'], monitor='val_acc', save_best_only=True, mode='max')
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
    ap.add_argument("-m", "--model", type=str, default="UNet.h5", help="path to output model")
    ap.add_argument("-p", "--plot", type=str, default="UNet.png",
                    help="path to output accuracy/loss plot")
    args = vars(ap.parse_args())
    return args


if __name__ == '__main__':
    args = args_parse()
    if not args['augment']:
        src_path = TRAIN_SRC_PATH
        lab_path = TRAIN_LAB_PATH
    train(args)
