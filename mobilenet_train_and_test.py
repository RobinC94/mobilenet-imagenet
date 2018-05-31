#!/usr/bin/python
import sys, os
from termcolor import cprint
from random import sample
import itertools

from keras.applications import imagenet_utils
from keras.applications.imagenet_utils import decode_predictions
# from keras.applications.imagenet_utils import preprocess_input
from keras.utils import to_categorical
from keras.applications import mobilenet
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from keras.layers.convolutional import Conv2D
from keras.optimizers import Adam, SGD, Adadelta
from keras.callbacks import ReduceLROnPlateau, CSVLogger, EarlyStopping, ModelCheckpoint, TensorBoard, LearningRateScheduler
from keras.metrics import top_k_categorical_accuracy
import numpy as np
import json
from math import sqrt
import xml.etree.ElementTree

##configuration parameters
alpha = 1.0
img_size = 224
img_parent_dir = "/home/crb/datasets/imageNet/ILSVRC2016/ILSVRC/Data/CLS-LOC/"  # sub_dir: train, val, test

pair_layers_num = 13
filter_size = 3

nb_epoch = 200
batch_size = 64
evaluating_batch_size = 96

##used in 3rd-party model function
class_parse_file = "./tmp/imagenet_class_index.json"
imagenet_utils.CLASS_INDEX = json.load(open(class_parse_file))
# used internally
debug_flag = False


## public API
def evaluate_model(model):
    nb_eval = 50000
    data_gen = evaluating_data_gen()
    model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=1e-4), metrics=['accuracy', acc_top5])
    res = model.evaluate_generator(generator=data_gen,
                                   steps=nb_eval / evaluating_batch_size,
                                   use_multiprocessing=True,
                                   workers=16,
                                   max_q_size=16)
    cprint("top1 acc:" + str(res[1]), "red")
    cprint("top5 acc:" + str(res[2]), "red")

def train_model(model, epoch = nb_epoch):
    model.compile(loss='categorical_crossentropy', optimizer=Adadelta(lr=lr_fine_tune_schedule(0), decay=0.0001), metrics=['accuracy', acc_top5])
    lr_scheduler = LearningRateScheduler(lr_fine_tune_schedule)
    lr_reducer = ReduceLROnPlateau(factor=np.sqrt(0.1), cooldown=0, patience=5, min_lr=0.5e-6)
    early_stopper = EarlyStopping(min_delta=0.001, patience=10)
    csv_logger = CSVLogger('./result/train_mobilenet_imagenet.csv')
    ckpt = ModelCheckpoint(filepath="./weights/mobilenet_train_weights.{epoch:02d}.h5", monitor='loss', save_best_only=True,
                           save_weights_only=True)
    model.fit_generator(generator=training_data_gen(),
                        steps_per_epoch=1281167 / batch_size,  # 1281167 is the number of training data we have
                        validation_data=evaluating_data_gen(),
                        validation_steps=50000 / batch_size,
                        epochs=epoch, verbose=1, max_q_size=32,
                        workers=12,
                        use_multiprocessing=True,
                        callbacks=[lr_reducer, early_stopper, csv_logger, ckpt, lr_scheduler])
    cprint("training is done\n", "yellow")



def fine_tune(model, epoch = nb_epoch):
    # compile model to make modification effect!!!
    model.compile(loss='categorical_crossentropy', optimizer=SGD(lr=lr_fine_tune_schedule(0), momentum=0.9, decay=0.0001), metrics=['accuracy', acc_top5])
    #model.compile(loss='categorical_crossentropy', optimizer=Adadelta(lr=lr_fine_tune_schedule(0)), metrics=['accuracy', acc_top5])
    # fine tune
    lr_scheduler = LearningRateScheduler(lr_fine_tune_schedule)
    lr_reducer = ReduceLROnPlateau(factor=np.sqrt(0.1), cooldown=0, patience=5, min_lr=0.5e-6)
    early_stopper = EarlyStopping(min_delta=0.001, patience=10)
    csv_logger = CSVLogger('./result/fine_tune_mobilenet_imagenet.512.csv')
    ckpt = ModelCheckpoint(filepath="./weights/mobilenet_fine_tune_weights.512.{epoch:02d}.h5", monitor='loss', save_best_only=True,
                           save_weights_only=True)
    tensorboard = TensorBoard(log_dir='./logs', histogram_freq=1, write_images=True)
    model.fit_generator(generator=training_data_gen(),
                        steps_per_epoch=1281167 / batch_size,  # 1281167 is the number of training data we have
                        validation_data=evaluating_data_gen(),
                        validation_steps=50000 / batch_size,
                        epochs=epoch, verbose=1, max_q_size=32,
                        workers=12,
                        use_multiprocessing=True,
                        callbacks=[lr_reducer, early_stopper, csv_logger, ckpt, lr_scheduler])
    cprint("fine tune is done\n", "yellow")


##private API
def acc_top5(y_true, y_pred):
    return top_k_categorical_accuracy(y_true, y_pred, k=5)

def training_data_gen():
    datagen = ImageDataGenerator(
        channel_shift_range=10,
        horizontal_flip=True,  # randomly flip images

        preprocessing_function=mobilenet.preprocess_input)

    img_dir = os.path.join(img_parent_dir, "train")
    img_generator = datagen.flow_from_directory(
        directory=img_dir,
        target_size=(img_size, img_size),
        color_mode="rgb",
        class_mode="categorical",
        batch_size=batch_size,
        shuffle=True)

    return img_generator


def evaluating_data_gen():
    datagen = ImageDataGenerator(
        preprocessing_function=mobilenet.preprocess_input)

    img_dir = os.path.join(img_parent_dir, "val")
    img_generator = datagen.flow_from_directory(
        directory=img_dir,
        target_size=(img_size, img_size),
        color_mode="rgb",
        class_mode="categorical",
        batch_size=evaluating_batch_size,
        shuffle=True)

    return img_generator


def generate_digit_indice_dict():
    digit_indice_dict = {value[0]: int(key) for key, value in imagenet_utils.CLASS_INDEX.items()}
    return digit_indice_dict

def lr_fine_tune_schedule(epoch):
    lr = 1e-4
    if epoch >= 7:
        lr *= sqrt(0.1)
    if epoch >= 5:
        lr *= sqrt(0.1)
    if epoch >= 3:
        lr *= sqrt(0.1)
    if epoch >= 1:
        lr *= sqrt(0.1)
    print('Learning rate: ', lr)
    return lr


def get_conv_layers_list(model):
    '''
        only  choose layers which is conv layer, and its filter_size must be same as param "filter_size"
    '''
    res = []
    layers = model.layers
    for i, l in enumerate(layers):
        if isinstance(l, Conv2D):
            if isinstance(l, mobilenet.DepthwiseConv2D) and l.depthwise_kernel.shape.as_list()[:2] == [filter_size,
                                                                                                       filter_size]:
                res += [i]
            elif l.kernel.shape.as_list()[:2] == [filter_size, filter_size]:
                res += [i]

    return res[:pair_layers_num]


# private data member
digit_indice_dict = generate_digit_indice_dict()

##for debug:
if __name__ == "__main__":

    debug_flag = False
    test = [5, 6]
    '''
    if 1 in test:
        # 1: check ImageDataGenerator's label is same as official defined,
        # data member "class_indices" contain the mapping info of ImageDataGenerator: digit_name:indice
        # official defined mapping file "imagenet_class_index.json": indice_str:[digit_name, string]
        official_mapping = imagenet_utils.CLASS_INDEX
        official_mapping = {value[0]: int(key) for key, value in official_mapping.iteritems()}
        # print official_mapping
        data_gen = training_data_gen()
        generator_mapping = data_gen.class_indices
        # print generator_mapping
        assert (official_mapping == generator_mapping)
        cprint("generator infered mapping is same as official defined, so ImageDataGenerator can be used", "green")
        data, label = data_gen.next()
        # print label.shape

    if 2 in test:
        # 2 check digit_indice_dict
        print digit_indice_dict

    if 3 in test:
        # 3 check evaluating data generator; note: digit string in image file name isn't equal to img's digit name is evaluation dataset
        data_gen = evaluating_data_gen()
        imgs, labels = data_gen.next()
        indice_list = labels.argmax(axis=1)
        cprint(imgs.shape, "red")
        cprint(labels.shape, "red")
        for i in indice_list:
            print imagenet_utils.CLASS_INDEX[str(i)][0]

    if 4 in test:
        # 4 check get_conv_layers_list
        weights_dir = "./weights"
        model = instanciate_mobilenet(weights_dir)
        print get_conv_layers_list(model)

    if 5 in test:
        # 5 test get_kernel_stack and set function; result before set and after set must be same
        weights_dir = "./weights"
        model = instanciate_mobilenet(weights_dir)
        img_path = "/media/flex/d/guizi/data/imageNet/2016/ILSVRC/Data/CLS-LOC/train/n03884397/n03884397_993.JPEG"
        res1 = evaluate_model1(model, img_path)

        conv_layers_list = get_conv_layers_list(model)
        kernels_stack = get_kernels_stack(model, conv_layers_list)
        set_modified_kernels_stack_to_model(model, kernels_stack, conv_layers_list)
        res2 = evaluate_model1(model, img_path)
        assert (res1 == res2)
        cprint("get kernel slice and set API ok", "green")

    if 6 in test:
        # check fine_tune
        fine_tune(model)
'''