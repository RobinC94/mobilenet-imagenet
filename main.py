#!/usr/bin/python
import sys, os
from termcolor import cprint

import keras

import mobilenet_modify as modi
import mobilenet_train_and_test as trte


#coonfig
modi.alpha = 1.0; alpha_str = "1_0"
modi.img_size = 224

modi.pair_layers_num = 14
modi.r_thresh = 0.94

trte.pair_layers_num = modi.pair_layers_num

def main():
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    weights_dir = "./weights"
    modi.print_config()
    model = modi.instanciate_mobilenet(weights_dir)
    #model.summary()
    keras.utils.plot_model(model, show_shapes=True)
    modi.print_conv_layer_info(model)

    #trte.evaluate_model2(model)
    #img_path = "/data1/datasets/imageNet/ILSVRC2016/ILSVRC/Data/CLS-LOC/train/n03884397/n03884397_993.JPEG"
    #trte.evaluate_model1(model, img_path)

    kmeans_k=256

    file = "./tmp/mobilenet_" + str(modi.pair_layers_num) + "_" + str(kmeans_k)
    modi.modify_model(model,k=kmeans_k,file_save=file)
    #modi.modified_model_from_file(model,file_load=file)

    trte.evaluate_model2(model)
    trte.fine_tune(model)

if __name__ == "__main__":
    main()

