# MNIST_Recognition

OOP implementation of a MLP. Uses elu activation functions, cross entropy with softmax (softmax taken out for debugging purposes at the moment). Processes each batch as a matrix. Truncated initialization of weights. 

Works, but is buggy (nans can appear). Pretty certain it is because of all the exponentiation, could probably be fixed with some well placed +1e-10 and/or np.nan_to_num.
