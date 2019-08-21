from keras.layers import *
from keras.models import Sequential
from keras.regularizers import l2

def Conv(filters=16, kernel_size=(3,3,3), activation='relu', input_shape=None):
    if input_shape:
        return Conv3D(filters=filters, kernel_size=kernel_size, padding='Same', 
                      activation=None, input_shape=input_shape,
                      kernel_initializer="he_normal")
#                       kernel_initializer="he_normal", kernel_regularizer=l2(1e-4))
    else:
        return Conv3D(filters=filters, kernel_size=kernel_size, padding='Same', 
                      activation=None,
                      kernel_initializer="he_normal")
#                       kernel_initializer="he_normal", kernel_regularizer=l2(1e-4))

def BatchNormalization_Relu(activation='relu'):
    return [BatchNormalization(), Activation(activation=activation)]

def BatchNormalization_Relu_MaxPooling3D(activation='relu'):
    return [BatchNormalization(), Activation(activation=activation), MaxPool3D()]

# Define Model
def CNN3D(input_dim, num_classes):
    model = Sequential([
        
        Conv(8, (3,3,3), input_shape=input_dim),
        *BatchNormalization_Relu(),
        Conv(16, (3,3,3)),
        *BatchNormalization_Relu_MaxPooling3D(),
        
        Conv(16, (3,3,3)),
        *BatchNormalization_Relu(),
        Conv(32, (3,3,3)),
        *BatchNormalization_Relu_MaxPooling3D(),
        
        Conv(32, (3,3,3)),
        *BatchNormalization_Relu(),
        Conv(64, (3,3,3)),
        *BatchNormalization_Relu_MaxPooling3D(),
        
        Conv(64, (3,3,3)),
        *BatchNormalization_Relu(),
        Conv(128, (3,3,3)),
        *BatchNormalization_Relu_MaxPooling3D(),
        
        Conv(128, (3,3,3)),
        *BatchNormalization_Relu(),
        Conv(256, (3,3,3)),
        *BatchNormalization_Relu(),
        Conv(512, (3,3,3)), # -11
        *BatchNormalization_Relu_MaxPooling3D(),
        
        Flatten(),
        Dense(1024, activation='relu'),
        Dropout(0.5),
        
        Dense(1024, activation='relu'),
        Dropout(0.5),
        Dense(num_classes),
        Activation(activation='softmax'),
        
    ])

    return model