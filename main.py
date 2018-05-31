#!/usr/bin/python
import os
import keras
from termcolor import cprint
from keras.backend.tensorflow_backend import set_session
import tensorflow as tf

from keras.applications import mobilenet
from keras.layers.convolutional import Conv2D

from mobilenet_modify import instantiate_mobilenet, modify_model, cluster_model_kernels, save_cluster_result, load_cluster_result
from mobilenet_train_and_test import evaluate_model, fine_tune, train_model
from mobilenet_new import  MobileNet_new


def print_config(alpha, img_size):
    cprint("alpha is: " + str(alpha), "red")
    cprint("image size is: " + str((img_size,) * 2), "red")

def print_conv_layer_info(model):
    f = open("./tmp/conv_layers_info.txt", "w")
    f.write("layer index   filter number   filter shape(HWCK)\n")
    for i, l in enumerate(model.layers):
        if isinstance(l, Conv2D):
            if isinstance(l, mobilenet.DepthwiseConv2D):
                print i, "DepthwiseConv2D", l.name, l.filters, l.depthwise_kernel.shape.as_list()
                print >> f, i, "DepthwiseConv2D", l.name, l.filters, l.depthwise_kernel.shape.as_list()
            else:
                print i, "Conv2D", l.name, l.filters, l.kernel.shape.as_list()
                print >> f, i, "Conv2D", l.name, l.filters, l.kernel.shape.as_list()

def main():
    os.environ['CUDA_VISIBLE_DEVICES'] = '3'
    config = tf.ConfigProto()
    #config.gpu_options.per_process_gpu_memory_fraction = 0.6
    set_session(tf.Session(config=config))

    alpha = 1
    img_size = 224

    print_config(alpha, img_size)
    model = instantiate_mobilenet(alpha, img_size)
    #model.summary()
    keras.utils.plot_model(model, show_shapes=True)
    print_conv_layer_info(model)

    #evaluate_model(model)
    #img_path = "/data1/datasets/imageNet/ILSVRC2016/ILSVRC/Data/CLS-LOC/train/n03884397/n03884397_993.JPEG"

    kmeans_k=512

    file = "./tmp/mobilenet_test_" + str(alpha) + "_" + str(img_size) + "_" + str(kmeans_k)

    #cluster_id, temp_kernels = cluster_model_kernels(model, k=kmeans_k, t=3)
    #save_cluster_result(cluster_id, temp_kernels, file)
    cluster_id, temp_kernels = load_cluster_result(file)

    #file = "./tmp/mobilenet_" + str(alpha) + "_" + str(img_size) + "_" + str(kmeans_k)
    #save_cluster_result(cluster_id, temp_kernels, file)

    model_new = modify_model(model, cluster_id, temp_kernels)

    evaluate_model(model_new)
    fine_tune(model_new, epoch=10)

    #train_model(model, epoch=10)

if __name__ == "__main__":
    main()

