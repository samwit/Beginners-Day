import re
import pickle
import keras.backend as K
from keras.preprocessing import image
from keras.models import Sequential
from keras.layers.core import Flatten, Dense, Dropout, Lambda
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.layers.pooling import GlobalAveragePooling2D
from keras.optimizers import SGD, RMSprop, Adam

import matplotlib.pyplot as plt
import collections
#import bcolz
from numpy.random import normal

def limit_gpu_mem():
    K.get_session().close()
    cfg = K.tf.ConfigProto()
    cfg.gpu_options.allow_growth = True
    K.set_session(K.tf.Session(config=cfg))

def dump(obj, fname): pickle.dump(obj, open(fname, 'wb'))
def load(fname): return pickle.load(open(fname, 'rb'))

def plot_train(hist):
    h = hist.history
    if 'acc' in h:
        meas='acc'
        loc='lower right'
    else:
        meas='loss'
        loc='upper right'
    plt.plot(hist.history[meas])
    plt.plot(hist.history['val_'+meas])
    plt.title('model '+meas)
    plt.ylabel(meas)
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc=loc)


def load_glove(loc):
    return (load_array(loc+'.dat'),
        pickle.load(open(loc+'_words.pkl','rb'), encoding='latin1'),
        pickle.load(open(loc+'_idx.pkl','rb'), encoding='latin1'))

# bcolz arrays

def load_array(fname): return bcolz.open(fname)[:]

def save_array(fname, arr):
    c=bcolz.carray(arr, rootdir=fname, mode='w')
    c.flush()

def get_batches( path, gen=image.ImageDataGenerator(), shuffle=True, batch_size=8, class_mode='categorical'):
    return gen.flow_from_directory(path, target_size=(299,299),
            class_mode=class_mode, shuffle=shuffle, batch_size=batch_size)
