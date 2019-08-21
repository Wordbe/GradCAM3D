import tensorflow as tf
import numpy as np
from keras import backend as K
from keras.models import Sequential, model_from_json, Model
from keras.layers import Lambda, Input
from tensorflow.python.framework import ops
from scipy.ndimage.interpolation import zoom
import keras
import tempfile
import os

def loss_calculation(x, category_index, nb_classes):
    return tf.multiply(x, K.one_hot((category_index), nb_classes))

def loss_calculation_shape(input_shape):
    return input_shape

def normalize(x):
    return x / (K.sqrt(K.mean(K.square(x))) + 1e-5)

def prepareGradCAM(input_model, conv_layer_idx, nb_classes):
    model = input_model

    #  because non-manufacturability is 1
    explanation_catagory = 1
    loss_function = lambda x: loss_calculation(x, explanation_catagory, nb_classes)
    model.add(Lambda(loss_function,
                     output_shape=loss_calculation_shape))

    #  use the loss from the layer before softmax. As best practices
    loss = K.sum(model.layers[-1].output)
    # last fully Convolutional layer to use for computing GradCAM
    conv_output = model.layers[conv_layer_idx].output
    
    grads = normalize(K.gradients(loss, conv_output)[0])
    gradient_function = K.function([model.layers[0].input, K.learning_phase()], 
                                   [conv_output, grads])
    return gradient_function


def registerGradient():
    if "GuidedBackProp" not in ops._gradient_registry._registry:
        @ops.RegisterGradient("GuidedBackProp")
        def _GuidedBackProp(op, grad):
            dtype = op.inputs[0].dtype
            return grad * tf.cast(grad > 0., dtype) * \
                tf.cast(op.inputs[0] > 0., dtype)

def modifyBackprop(model, name, ld_model_fn, input_shape, nb_classes, weight_name):
    registerGradient()
    g = tf.get_default_graph()
    with g.gradient_override_map({'Relu': name}):

        # get layers that have an activation
        layer_dict = [layer for layer in model.layers[1:]
                      if hasattr(layer, 'activation')]

        # replace relu activation
        for layer in layer_dict:
            if layer.activation == keras.activations.relu:
                layer.activation = tf.nn.relu

    model = ld_model_fn(input_shape, nb_classes)
    model.load_weights(weight_name)
    
    #   Popping the softmax layer as it creates ambiguity in the explanation
    model.pop()
    return model
        
def compileSaliencyFunction(model, ld_model_fn, input_shape, nb_classes, weight_name, activation_layer=-9):
    guidedModel = modifyBackprop(model, 'GuidedBackProp', ld_model_fn, input_shape, nb_classes, weight_name)
    
    input_img = guidedModel.input
    layer_output = guidedModel.layers[activation_layer].output
    saliency = K.gradients(K.sum(layer_output), input_img)[0]
    
    return K.function([input_img, K.learning_phase()], [saliency])


def GradCAM(gradient_function, input_file):
    # Last conv output, and grad value
    output, grads_val = gradient_function([input_file, 0])
    grads_val = grads_val / (np.max(grads_val) + K.epsilon())
    
    weights = np.mean(grads_val, axis=(1, 2, 3))
    weights.flatten()
    
    if K.image_data_format() == "channels_last":
        grad_cam = np.ones(output.shape[1:-1], dtype=K.floatx())
    else:
        grad_cam = np.ones(output.shape[2:], dtype=K.floatx())

    for i, w in enumerate(np.transpose(weights)):
        if K.image_data_format() == "channels_last":
            grad_cam += w * output[0, ..., i]
        else:
            grad_cam += w * output[0, i, ...]
    
    # Post process of the heatmap
    grad_cam = np.maximum(grad_cam, 0)
    grad_cam = grad_cam / np.max(grad_cam)
    
    # attention map
    attMap = np.zeros_like(input_file)
    
    zoom_factor = [i / (j * 1.0) for i, j in iter(zip(input_file.shape[1:-1], grad_cam.shape))]
    attMap[0,..., 0] = zoom(grad_cam, zoom_factor)

    attMap = (1 * np.float32(attMap)) + (1 * np.float32(input_file))
    attMap = (attMap / np.max(attMap))
    return attMap


