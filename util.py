import os
import numpy as np

def get_mnist(data_dir, onehot=False, scale01=True):
    """
    Load MNIST dataset from data_dir
    Returns a tuple of four numpy arrays (trX, teX, trY, teY)
    where tr and te mean train and test, respectively.
    X arrays (inputs) have shape (N, 28**2)
    Y arrays (labels) have shape (N,) if onehot=False, else (N, 10) 
    """
    fd = open(os.path.join(data_dir,'train-images-idx3-ubyte'))
    loaded = np.fromfile(file=fd,dtype=np.uint8)
    _trX = loaded[16:].reshape((60000,28*28)).astype('float32')

    fd = open(os.path.join(data_dir,'train-labels-idx1-ubyte'))
    loaded = np.fromfile(file=fd,dtype=np.uint8)
    _trY = loaded[8:].reshape((60000))

    fd = open(os.path.join(data_dir,'t10k-images-idx3-ubyte'))
    loaded = np.fromfile(file=fd,dtype=np.uint8)
    _teX = loaded[16:].reshape((10000,28*28)).astype('float32')

    fd = open(os.path.join(data_dir,'t10k-labels-idx1-ubyte'))
    loaded = np.fromfile(file=fd,dtype=np.uint8)
    _teY = loaded[8:].reshape((10000))
    
    _trY = np.asarray(_trY)
    _teY = np.asarray(_teY)

    if onehot:
        trY = np.zeros((_trY.shape[0], 10), dtype='float32')
        trY[np.arange(_trY.shape[0]), _trY] = 1
        teY = np.zeros((_teY.shape[0], 10), dtype='float32')
        teY[np.arange(_teY.shape[0]), _teY] = 1
    else:
        trY = _trY
        teY = _teY

    if scale01:
        trX = (_trX / 255.0).astype('float32')
        teX = (_teX / 255.0).astype('float32')
    else:
        trX = _trX
        teX = _teX

    return trX, teX, trY, teY
