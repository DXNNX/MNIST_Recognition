from simple_network import Network
import numpy as np
import h5py
import matplotlib.pyplot as plt
from matplotlib import style
style.use('ggplot')

def get_data():
    data_dir = "data\\"
    dtr = h5py.File(data_dir+"data_train.hdf5",'r')
    dte = h5py.File(data_dir+"data_test.hdf5",'r')
    return dtr['feats'][()], dtr['targs'][()], dte['feats'][()], dte['targs'][()]

def main():

    feats, targs, f_test, t_test = get_data()
    print(feats.shape, f_test.shape)
    f,t = feats, targs
    NN = Network([784,30,30,10])

    NN.train(f, t, 10, 200, 1e-1) 

    print(NN.evaluate(f,t))


main()

