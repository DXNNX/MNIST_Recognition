import numpy as np
import time
import matplotlib.pyplot as plt

def trunc_norm(bnd, shape):
    tmp = np.random.randn(*shape)
    mask = np.abs(tmp)>bnd
    tmp[mask] = np.random.uniform(low = -bnd, high=bnd, size=shape)[mask]
    return tmp

def elu(x):
    smaller_zero_inds = x<0
    act = x
    act[smaller_zero_inds] = (np.exp(x)-1)[smaller_zero_inds]
    return act
    
def elu_prime(x):
    smaller_zero_inds = x<0
    d_elu = np.ones_like(x, dtype = float)
    d_elu[smaller_zero_inds] = np.exp(x)[smaller_zero_inds]
    return d_elu


class Network:

    def __init__(self, shape):
        self.weights =[trunc_norm(0.01, (neu_from, neu_to)) 
                                  for neu_from,neu_to in zip(shape[:-1],shape[1:])]

        self.biases = [0.01*np.ones((col,)) 
                                  for col in shape[1:]]

        self.act_fct = elu

        self.shape = shape
        self.num_layers = len(shape)

    def output(self,a):
        for b,w in zip(self.biases[:-1],self.weights[:-1]):
            a=self.act_fct(a.dot(w) + b)
        a=softmax(a.dot(self.weights[-1]) + self.biases[-1])
        #a=self.act_fct(a.dot(self.weights[-1]) + self.biases[-1])
        return a


    def train(self, feats, targs, epochs, batch_size, eta, val_data=False, f_test=None, t_test=None):

        len_data = len(feats)
        ct=1
        for i in range(epochs):
            p = np.random.permutation(len_data)
            feats, targs = feats[p], targs[p]
            for start in range(0,len_data, batch_size):
                end = start + batch_size
                batch_f, batch_t = feats[start:end], targs[start:end]
                self.update_batch(batch_f, batch_t, eta, ct)

            if val_data:
                print("Epoch {0}: {1}".format(
                    i,self.evaluate(f_test, t_test)))
            else:
                print("Epoch {0}: {1}".format(
                    i,self.evaluate(feats, targs)))
                print(self.cross_ent(feats, targs))
    
    def update_batch(self, batch_f, batch_t, eta, ct):
        len_batch = len(batch_f)

        del_b = [np.zeros(b.shape) for b in self.biases]
        del_w = [np.zeros(w.shape) for w in self.weights]
        
        del_w,del_b=self.backprop_algo(batch_f, batch_t, len_batch, ct)

        self.weights=[w-(eta/len_batch) * dw for w,dw in zip(self.weights,del_w)]      
        self.biases=[b-(eta/len_batch) * db for b,db in zip(self.biases,del_b)]
            
    def backprop_algo(self, batch_f, batch_t, len_batch, ct):

        # Input.
        act = [np.zeros((len_batch, size)) for size in self.shape]
        z = [np.zeros((len_batch, size)) for size in self.shape[1:]]
        act[0] = batch_f
        y = batch_t

        # Forward.
        for i in range(self.num_layers-2):
            z[i] = act[i].dot(self.weights[i]) + self.biases[i]
            act[i+1] = self.act_fct(z[i])
        z[-1] = act[-2].dot(self.weights[-1]) +self.biases[-1]
        act[-1] = softmax(z[-1])
        
        # Backward.
        delta = [np.zeros((len_batch, size)) for size in self.shape[1:]]
        delta[-1] = act[-1]-y

        for l in range(2,self.num_layers):
            delta[-l] = delta[-(l-1)].dot(self.weights[-(l-1)].T)*elu_prime(z[-l])

        # For a single neuron take sample activations and times with their corresponding deltas (of a neuron), 
        # add the contributions up. Repeat for the deltas and activations of other neurons.
        del_w = [act[i].T.dot(delta[i]) for i in range(self.num_layers-1)]
        
        # All bias derivatves added up.
        delta = [np.sum(delta[i], axis=0) for i in range(self.num_layers-1)]

        return del_w,delta
     
    def evaluate(self, f_test, t_test):
        test_results = np.argmax(self.output(f_test), axis=1) == np.argmax(t_test, axis=1)
        return np.mean(test_results)

    def cross_ent(self, f, t):
        return np.mean(np.sum(-np.multiply(t,np.log(self.output(f))), axis=1))


 

def softmax(x):
    tmp = np.exp(x)
    return (tmp.T/np.sum(tmp, axis=1)).T

#def softmax_prime(x):
#    out = np.zeros_like(x, dtype = float)
#    denom = np.sum(np.exp(x), axis=1)**2
#    saved_mask = np.ones_like(x, dtype = float)

#    # Have to calculate explicitly for each neuron because the result is dependent on 
#    # not only the neuron being derivated but also the neuron's (in the layer) that are 
#    # being regarded as a constant, resulting in the formula changing for each one.
#    # Transpose to change shape (sample,neuron) -> (neuron,sample) and allow easy enumeration.
#    for i, arr in enumerate(x.T):
#        mask = np.copy(saved_mask)
#        mask[:,i] = 0

#        num = np.exp(arr)*np.sum(np.multiply(np.exp(x), mask), axis=1)
        
#        if np.invert(np.isfinite(num)).any() or np.invert(np.isfinite(denom)).any():
#            print("asd")
#        out[:,i] = num/denom 
         
#    return out