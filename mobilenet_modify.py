#!/usr/bin/python
import sys, os
from termcolor import cprint
import scipy.stats as stats
import keras
from array import array

from keras.applications import imagenet_utils
from keras.applications import MobileNet
from keras.applications import mobilenet
from keras.applications.mobilenet import DepthwiseConv2D
from mobilenet_new import MobileNet_new

from keras.layers.convolutional import Conv2D
from keras.layers.normalization import BatchNormalization
from keras.layers import Dense
import numpy as np
import json
from Bio import Cluster

#############################
##configuration parameters
img_parent_dir = "/home/crb/datasets/imageNet/ILSVRC2016/ILSVRC/Data/CLS-LOC/"  # sub_dir: train, val, test
filter_size = 3

kmeans_k=256

##used in 3rd-party model function
class_parse_file = "./tmp/imagenet_class_index.json"
imagenet_utils.CLASS_INDEX = json.load(open(class_parse_file))
# used internally
debug_flag = False

#####################################
## public API
def instantiate_mobilenet(alpha, img_size):
    if alpha == 1.0:
        alpha_str = "1_0"
    elif alpha == 0.75:
        alpha_str = "7_5"
    elif alpha == 0.5:
        alpha_str = "5_0"
    elif alpha == 0.25:
        alpha_str = "2_5"
    else:
        raise ValueError('alpha wrong.')

    model = MobileNet(input_shape=(img_size, img_size, 3), alpha=alpha, include_top=True, weights=None)
    model.load_weights("weights/mobilenet_train_weights_69.79.h5")
    #model.load_weights("/home/crb/Desktop/mobilenet_1_0_224_tf_1.h5")

    return model

def cluster_model_kernels(model, k=kmeans_k, t = 1):
    # 1 get 3x3 conv layers
    conv_layers_list = get_conv_layers_list(model)
    cprint("selected conv layers is:" + str(conv_layers_list), "red")

    # 2 get kernels array
    kernels_array = get_kernels_array(model, conv_layers_list)
    print "num of kernels:" + str(np.shape(kernels_array)[0])

    # 3 get clusterid and temp
    cluster_id, temp_kernels = cluster_kernels(kernels_array, k=k, times=t)

    return cluster_id, temp_kernels

def modify_model(model, cluster_id, temp_kernels):
    # 1 get 3x3 conv layers
    conv_layers_list = get_conv_layers_list(model)
    cprint("selected conv layers is:" + str(conv_layers_list), "red")

    # 2 get kernels array
    kernels_array = get_kernels_array(model, conv_layers_list)
    print "num of kernels:" + str(np.shape(kernels_array)[0])

    # 3 get coefficient a
    coef_a, coef_b = get_coefficients(kernels_array, cluster_id, temp_kernels)

    model_new = MobileNet_new(input_shape=(224, 224, 3), alpha=1.0, include_top=True, weights=None)

    # 4 set model weights
    set_cluster_weights_to_old_model(model, cluster_id,temp_kernels,coef_a,coef_b,conv_layers_list)

    set_cluster_weights_to_model(model, model_new,
                                 coef_a=coef_a,
                                 coef_b=coef_b,
                                 clusterid=cluster_id,
                                 cdata=temp_kernels,
                                 conv_layers_list=conv_layers_list)

    return model_new

def save_cluster_result(clusterid, temp, f):
    f_clusterid = f + "_clusterid.npy"
    f_temp = f + "_temp.npy"
    np.save(f_clusterid, clusterid)
    np.save(f_temp, temp)

def load_cluster_result(f):
    f_clusterid = f + "_clusterid.npy"
    f_temp = f + "_temp.npy"
    clusterid = np.load(f_clusterid)
    temp = np.load(f_temp)

    print 'loading cluster result done.'
    return clusterid, temp


######################################
##private API
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

    return res

def get_kernels_array(model, conv_layers_list):
    kernels_num = 0

    kernels_buf=array('d')
    for l in conv_layers_list:
        weights = model.layers[l].get_weights()[0]  ##0 weights, 1 bias; HWCN
        for i in range(weights.shape[-1]):  ##kernel num
            for s in range(weights.shape[-2]):  # kernel depth
                weights_slice = weights[:, :, s, i]  # HWCK
                for w in weights_slice.flatten():
                    kernels_buf.append(w)
                kernels_num+=1

    kernels_array = np.frombuffer(kernels_buf,dtype=np.float).reshape(kernels_num,filter_size**2)
    return kernels_array

def get_weighted_layers_list(model):
    '''
        get all layers with weights without conv3x3
    '''
    res = []
    layers = model.layers
    for i,l in enumerate(layers):
        if (isinstance(l,Conv2D)or isinstance(l, BatchNormalization) or isinstance(l, Dense)):
            res += [i]
    return res

def least_square(dataa, datab):
    assert (dataa.shape == datab.shape)
    dataa = dataa.reshape(-1)
    datab = datab.reshape(-1)
    a, b = np.polyfit(dataa, datab, 1)  ##notice the direction: datab = a*dataa + b
    return (a, b)

def cluster_kernels(kernels_array, k=kmeans_k, times=1):
    print "start clustering"

    clusterid = []
    error_best = float('inf')
    for i in range(times):
        clusterid_single, error, nfound = Cluster.kcluster(kernels_array, nclusters=k, dist='a')
        if error < error_best:
            clusterid = clusterid_single
            error_best = error
    print 'error:', error_best

    cdata, cmask = Cluster.clustercentroids(kernels_array, clusterid=clusterid, )

    print "end clustering"

    return clusterid, cdata

def get_coefficients(kernels_array, clusterid, cdata):
    kernels_num = np.shape(kernels_array)[0]
    coef_a = np.zeros(kernels_num)
    coef_b = np.zeros(kernels_num)

    avg_sum = 0
    for i in range(kernels_num):
        cent_id = clusterid[i]
        kernel = kernels_array[i]
        cent = cdata[cent_id]
        a, b = least_square(cent, kernel)
        coef_a[i] = a
        coef_b[i] = b
        r = abs(stats.pearsonr(kernel, cent)[0])
        avg_sum += r
    avg = avg_sum / kernels_num

    print "average r2:%6.4f" % (avg)

    return coef_a, coef_b

def generate_digit_indice_dict():
    digit_indice_dict = {value[0]: int(key) for key, value in imagenet_utils.CLASS_INDEX.items()}
    return digit_indice_dict

def set_cluster_weights_to_model(model, model_new, clusterid, cdata, coef_a, coef_b,conv_layers_list):
    kernel_id = 0
    for l in conv_layers_list:
        if isinstance(model.layers[l],DepthwiseConv2D):
            weights_new = model_new.layers[l].get_weights()
            i=0
            for s in range(model.layers[l].input_shape[-1]):  ##kernel num
                weights_new[0][s,i] = coef_a[kernel_id]
                weights_new[1][s,i] = coef_b[kernel_id]
                cent_id = clusterid[kernel_id]
                cent = cdata[cent_id]
                weights_new[2][:,:,s,i]=np.array(cent).reshape(filter_size, filter_size)  # HWCK
                kernel_id += 1

        elif isinstance(model.layers[l],Conv2D):
            weights_new = model_new.layers[l].get_weights()
            for i in range(model.layers[l].filters):  ##kernel num
                for s in range(model.layers[l].input_shape[-1]):  # kernel depth
                    weights_new[0][s,i] = coef_a[kernel_id]
                    weights_new[1][s,i] = coef_b[kernel_id]
                    cent_id = clusterid[kernel_id]
                    cent = cdata[cent_id]
                    weights_new[2][:,:,s,i]=np.array(cent).reshape(filter_size, filter_size)  # HWCK
                    kernel_id += 1

        model_new.layers[l].set_weights(weights_new)

    weighted_layers_list = get_weighted_layers_list(model)
    for l in weighted_layers_list:
        if l not in conv_layers_list:
            weights = model.layers[l].get_weights()
            model_new.layers[l].set_weights(weights)

def set_cluster_weights_to_old_model(model, clusterid, cdata, coef_a, coef_b, conv_layers_list):
    kernels_id = 0
    for l in conv_layers_list:
        weights = model.layers[l].get_weights()
        for i in range(weights[0].shape[-1]):  ##kernel num
            for s in range(weights[0].shape[-2]):  # kernel depth
                cent=clusterid[kernels_id]
                temp=cdata[cent]
                a=coef_a[kernels_id]
                b=coef_b[kernels_id]
                weights[0][:, :, s, i] = np.array(a*temp+b).reshape(filter_size, filter_size)
                kernels_id += 1
        model.layers[l].set_weights(weights)

# private data member
digit_indice_dict = generate_digit_indice_dict()

##for debug:
if __name__ == "__main__":
    os.environ['CUDA_VISIBLE_DEVICES'] = '2'
    alpha = 1
    img_size = 224
    model = instantiate_mobilenet(alpha, img_size)

    print get_conv_layers_list(model)


