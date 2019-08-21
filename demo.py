import os
import numpy as np
import matplotlib.pyplot as plt
import keras
from math import ceil

from dataload import *

from keras.optimizers import Adam
from keras.losses import categorical_crossentropy
from conv3DNet import CNN3D

from GradCAM3D.GradCAM import prepareGradCAM, GradCAM

# GPU setting
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

CLASSES = ['large+abnormal', 'normal']
NUM_CLASS = len(CLASSES)
BATCH_SIZE = 1
IMG_DEPTH = 32
IMG_HEIGHT = 128
IMG_WIDTH = 128
NUM_CHANNEL = 1
NUM_EPOCH = 200
learning_rate = 1e-4
learning_rate_str = '1e-4'
data_src_dir = '/data/Humerus_tendon/maskdata2/'
weight_name = 'weights/Classification_5convslayers_CNN3D_batch1_lr1e-4_128x128.h5'
last_conv_layer_index = -10

def main():
    # Prepare data
    x_train, y_train, x_valid, y_valid, x_test, y_test = splitdata(data_src_dir, CLASSES)
    
    # Load model
    # input : (batch_size, conv_dim1, conv_dim2, conv_dim3, input_channels)
    input_shape = (IMG_DEPTH, IMG_HEIGHT, IMG_WIDTH, NUM_CHANNEL)
    model = CNN3D(input_shape, NUM_CLASS)
    model.compile(optimizer=Adam(lr=learning_rate, clipnorm=.001), 
                  loss=categorical_crossentropy, 
                  metrics=['accuracy'])
    
    # Grad-CAM
    # Remove the last activation layer
    model.pop()
    
    activation_function = prepareGradCAM(model, last_conv_layer_index, NUM_CLASS)
    saliency_fn = compileSaliencyFunction(model, CNN3D, input_shape, NUM_CLASS, weight_name,
                                          activation_layer=last_conv_layer_index)
    
    BATCH_SIZE = 1 # for Grad-CAM test
    num_step = ceil(len(x_test) / BATCH_SIZE)
    for i in range(num_step):
        start_index = i * BATCH_SIZE
        end_index = (i + 1) * BATCH_SIZE
        x_batch, y_batch = load_data(x_test[start_index:end_index],
                                     y_test[start_index:end_index],
                                     'test')
        
        fullfilename = x_test[start_index:end_index][0]
        filename = fullfilename.split('/')[-1]

        predicted_class = model.predict(x_batch)
        print(filename, 'has a predicted class', predicted_class)

        attMap = GradCAM(activation_function, x_batch)
        gBackprop = saliency_fn([x_batch, 0])
        gGradCam = gBackprop[0] * attMap
        gGradCam = (gGradCam / np.max(gGradCam))
        finalOutput = (1 * np.float32(gGradCam)) + 1*np.float32(x_batch)
        finalOutput = (finalOutput / np.max(finalOutput))
        finalOutput *= 255.0
        finalOutput = (finalOutput.squeeze()).astype('uint8')
        
        sitkImg = sitk.GetImageFromArray(finalOutput)
        sitk.WriteImage(sitkImg, 'GradCAM_outputs/results/' + filename)
        
        img = sitk.GetArrayFromImage(sitk.ReadImage(fullfilename))
        img = resize(img, IMG_DEPTH, IMG_HEIGHT, IMG_WIDTH)
        img = sitk.GetImageFromArray(img)
        sitk.WriteImage(mask, 'GradCAM_outputs/raw_image/' + filename)
        print(filename, 'saved!')
    
if __name__ == '__main__':
    main()
    
    