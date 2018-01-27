#!/usr/bin/python
import sys, os
from termcolor import cprint
import scipy.stats as stats
import keras

from keras.applications import imagenet_utils
from keras.applications import MobileNet
from keras.applications import mobilenet
from keras.layers.convolutional import Conv2D
import numpy as np
import json
from Bio import Cluster

#############################
##configuration parameters
alpha = 1.0
img_size = 224
img_parent_dir = "/data1/datasets/imageNet/ILSVRC2012/Data/"  # sub_dir: train, val, test

pair_layers_num = 1
filter_size = 3
r_thresh = 0.8

kmeans_k=256
batch_size = 32

##used in 3rd-party model function
class_parse_file = "./tmp/imagenet_class_index.json"
imagenet_utils.CLASS_INDEX = json.load(open(class_parse_file))
# used internally
debug_flag = False

#####################################
## public API
def print_config():
    cprint("alpha is: " + str(alpha), "red")
    cprint("image size is: " + str((img_size,) * 2), "red")
    cprint("image dir is: " + img_parent_dir, "red")
    cprint("batch size is: " + str(batch_size), "red")

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


def instanciate_mobilenet(weights_dir):
    if alpha == 1.0:
        alpha_str = "1_0"
    elif alpha == 0.75:
        alpha_str = "7_5"
    elif alpha == 0.5:
        alpha_str = "5_0"
    elif alpha == 0.25:
        alpha_str = "2_5"
    weights_file_name = os.path.join(weights_dir, "mobilenet_" + alpha_str + "_" + str(img_size) + "_tf.h5")
    #weights_file_name = "fine_tune_weights.h5"

    model = MobileNet(input_shape=(img_size, img_size, 3), alpha=alpha, include_top=True, weights=None)
    model.load_weights(weights_file_name)

    return model

def modify_model(model, k=kmeans_k,file_save = None):

    # 1 select conv layers
    conv_layers_list = get_conv_layers_list(model)
    cprint("selected conv layers is:" + str(conv_layers_list), "red")

    # 2 get kernels stack
    kernels_stack = get_kernels_stack(model, conv_layers_list)
    print "num of searched kernels:" + str(len(kernels_stack.keys()))

    modify_kernels(kernels_stack,k=k,f_save=file_save)
    set_modified_kernels_stack_to_model(model, kernels_stack, conv_layers_list)

def modified_model_from_file(model, file_load = None):

    # 1 select conv layers
    conv_layers_list = get_conv_layers_list(model)
    cprint("selected conv layers is:" + str(conv_layers_list), "red")

    # 2 get kernels stack
    kernels_stack = get_kernels_stack(model, conv_layers_list)
    print "num of searched kernels:" + str(len(kernels_stack.keys()))

    # 4 modify model
    modify_kernels_from_file(kernels_stack, f_load=file_load)
    set_modified_kernels_stack_to_model(model, kernels_stack, conv_layers_list)


######################################
##private API
def get_kernels_stack(model, conv_layers_list):
    kernels = []
    index = []
    zero_num = 0
    for l in conv_layers_list:
        weights = model.layers[l].get_weights()[0]  ##0 weights, 1 bias; HWCK
        for i in range(weights.shape[-1]):  ##kernel num
            for s in range(weights.shape[-2]):  # kernel depth
                weights_slice = weights[:, :, s, i]  # HWCK
                kernels += [(weights_slice)]
                index += [(l, i, s)]
                if np.allclose(weights_slice, 0.0, atol=1e-5):
                    zero_num += 1

    print "num of kernel slices that are close to 0: ", zero_num
    kernels_stack = {key: value for key, value in zip(index, kernels)}
    return kernels_stack

def least_square(dataa, datab):
    assert (dataa.shape == datab.shape)
    dataa = dataa.reshape(-1)
    datab = datab.reshape(-1)
    a, b = np.polyfit(dataa, datab, 1)  ##notice the direction: datab = a*dataa + b
    return (a, b)

def modify_kernels(kernels_stack, k=kmeans_k,f_save=None):
    kernels_keys=kernels_stack.keys()
    kernels_num = len(kernels_keys)
    kernels_array=np.zeros((kernels_num,filter_size**2))

    for i in range(kernels_num):
        kernel_id=kernels_keys[i]
        kernels_array[i]=kernels_stack[kernel_id].flatten()

    print "start clustering"

    clusterid,cdata,avg_r=cluster_kernels(kernels_array,k,10)

    print "end clustering"

    for i in range(kernels_num):
        cent_id = clusterid[i]
        kernel = kernels_array[i]
        cent = cdata[cent_id]
        a, b= least_square(cent, kernel)
        kernel_id=kernels_keys[i]
        kernels_stack[kernel_id]=a*cent.reshape(filter_size,filter_size)+b

    print "average r2: %6.4f\t" % (avg_r)

    if f_save != None:
        f_clusterid = f_save + "_clusterid.npy"
        f_cdata = f_save + "_cdata.npy"
        np.save(f_clusterid, clusterid)
        np.save(f_cdata, cdata)

def cluster_kernels(kernels_array, k=kmeans_k,times=1):
    n=np.shape(kernels_array)[0]
    best_r=0
    for i in range(times):
        clusterid, error, nfound = Cluster.kcluster(kernels_array, nclusters=k, dist='a')
        cdata, cmask = Cluster.clustercentroids(kernels_array, clusterid=clusterid, )
        avg_r=0
        for j in range(n):
            cent_id = clusterid[j]
            kernel = kernels_array[j]
            cent = cdata[cent_id]
            r = abs(stats.pearsonr(kernel, cent)[0])
            avg_r += r / n
        if avg_r>best_r:
            best_cluster=clusterid
            best_cdata=cdata
            best_r=avg_r
    return best_cluster,best_cdata,best_r


def modify_kernels_from_file(kernels_stack, f_load=None):
    kernels_keys = kernels_stack.keys()
    kernels_num = len(kernels_keys)
    kernels_array = np.zeros((kernels_num, filter_size ** 2))

    for i in range(kernels_num):
        kernel_id = kernels_keys[i]
        kernels_array[i] = kernels_stack[kernel_id].flatten()

    try:
        f_clusterid=f_load +"_clusterid.npy"
        f_cdata= f_load+"_cdata.npy"
        clusterid=np.load(f_clusterid)
        cdata=np.load(f_cdata)
        print "loading file done"
    except:
        print "cannot open file"
        sys.exit(0)

    avg_r=0
    for i in range(kernels_num):
        cent_id = clusterid[i]
        kernel = kernels_array[i]
        cent = cdata[cent_id]
        a, b= least_square(cent, kernel)
        r=abs(stats.pearsonr(kernel,cent)[0])
        avg_r += r/kernels_num
        kernel_id=kernels_keys[i]
        kernels_stack[kernel_id]=a*cent.reshape(filter_size,filter_size)+b

    print "average r2: %6.4f\t" % (avg_r)


def generate_digit_indice_dict():
    digit_indice_dict = {value[0]: int(key) for key, value in imagenet_utils.CLASS_INDEX.items()}
    return digit_indice_dict

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


def set_modified_kernels_stack_to_model(model, kernels_stack, conv_layers_list):
    for l in conv_layers_list:
        weights = model.layers[l].get_weights()
        for i in range(weights[0].shape[-1]):  ##kernel num
            for s in range(weights[0].shape[-2]):  # kernel depth
                weights[0][:, :, s, i] = kernels_stack[(l, i, s)]
        model.layers[l].set_weights(weights)

# private data member
digit_indice_dict = generate_digit_indice_dict()

##for debug:
if __name__ == "__main__":
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    weights_dir = "./weights"
    print_config()
    model = instanciate_mobilenet(weights_dir)
    # model.summary()
    keras.utils.plot_model(model, show_shapes=True)
    print_conv_layer_info(model)

    pair_layers_num = 14

    conv_layers_list = get_conv_layers_list(model)
    cprint("selected conv layers is:" + str(conv_layers_list), "red")
    kernels_stack = get_kernels_stack(model, conv_layers_list)
    print "num of searched kernels:" + str(len(kernels_stack.keys()))


