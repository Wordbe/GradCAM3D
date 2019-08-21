# Divide into train, validation, test set
import re, glob, random
import numpy as np
from math import ceil
from scipy import ndimage
import SimpleITK as sitk
from keras.utils import to_categorical

def tryint(s):
    try:
        return int(s)
    except:
        return s     
def alphanum_key(s):
    """ Turn a string into a list of string and number chunks.
        "z23a" -> ["z", 23, "a"]
    """
    return [ tryint(c) for c in re.split('([0-9]+)', s) ]
def sort_nicely(l):
    """ Sort the given list in the way that humans expect.
    """
    l.sort(key=alphanum_key)
    return l

def print_list_pair(x_list, y_list):
    print("Total : ", len(x_list))
    for i in range(len(x_list)):
        print(x_list[i], y_list[i])

def make_classlabel(x_list):
    y_list = []
    for x in x_list:
        for i in range(NUM_CLASS):
            if CLASSES[i] in x:
                y_list.append(i)
                break
    return y_list

def splitdata(data_src, CLASSES):
    x_ab = sort_nicely(glob.glob(data_src + CLASSES[0] + '/*')) # abnormal
    x_no = sort_nicely(glob.glob(data_src + CLASSES[1] + '/*')) # normal
    
    num_ab_sample = ceil(len(x_ab) / 10)
    num_no_sample = ceil(len(x_no) / 10)
    random.seed(0)

    x_valid_ab = random.sample(x_ab, num_ab_sample)
    x_valid_no = random.sample(x_no, num_no_sample)
    x_valid = x_valid_ab + x_valid_no
    x_ab = list(set(x_ab) - set(x_valid_ab))
    x_no = list(set(x_no) - set(x_valid_no))

    x_test_ab = random.sample(x_ab, num_ab_sample)
    x_test_no = random.sample(x_no, num_no_sample)
    x_test = x_test_ab + x_test_no
    x_ab = list(set(x_ab) - set(x_test_ab))
    x_no = list(set(x_no) - set(x_test_no))

    x_train = x_ab + x_no
#     print("x_train: {}\nx_valid: {}\nx_test: {}"
#           .format(len(x_train), len(x_valid), len(x_test)))
    y_train = make_classlabel(x_train)
    y_valid = make_classlabel(x_valid)
    y_test = make_classlabel(x_test)
    
#     print_list_pair(x_valid, y_valid)
    return map(sorted, [x_train, y_train, x_valid, y_valid, x_test, y_test])

def preprocess_img(img):  # uint8 to 0-1
    b = np.percentile(img, 99)
    t = np.percentile(img, 1)
    img = np.clip(img, t, b)
    img= (img - b) / (t-b)
    img= 1-img
    return img

def resize(data, img_dep=180., img_rows=180., img_cols=180.,mode='constant'):
   resize_factor = (img_dep / data.shape[0], img_rows / data.shape[1], img_cols / data.shape[2])
   data = ndimage.zoom(data, resize_factor, order=0, mode=mode, cval=0.0)
   return data

def load_data(x_batch, y_batch, phase='train', pick_label=None):
    batch_size = len(x_batch)
    batch_img = []
    batch_label = []
        
    for i in range(batch_size):
        image_ = sitk.ReadImage(x_batch[i])
        
        if phase == 'train':
            # Augmentation
            tx = bspline_tranform_parameter(image_)
            image_ = bspline_tranform(image_,tx,sitk.sitkNearestNeighbor)
            
        image_ = sitk.GetArrayFromImage(image_)
        
        # if you contain a specific classes from image(mask)
        if pick_label:
            # Pick specific label
            image_[image_ != pick_label] = 0
            image_[image_ == pick_label] = 1
        
        # Preprocess (input range -> 0~1)
        image_ = preprocess_img(image_)
        # Resize
        image_ = resize(image_, IMG_DEPTH, IMG_HEIGHT, IMG_WIDTH)
        image = np.reshape(image_, (IMG_DEPTH, IMG_HEIGHT, IMG_WIDTH, 1))
        batch_img.append(image)
        
        batch_label.append(y_batch[i])
        
    return np.array(batch_img), np.array(to_categorical(batch_label, 
                                                        num_classes=NUM_CLASS))
    