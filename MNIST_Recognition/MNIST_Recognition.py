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
    f,t = feats[:10], targs[:10]
    NN = Network([784,30,30,10])

    NN.train(f, t, 100, 100, 1e-2) 

    print(NN.evaluate(f,t))


main()

