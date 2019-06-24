import numpy as np
from typing import * 
import sys

class Shape():     

    def __init__(self, dim): 
        self.dimension = []
        for i in range(len(dim)): 
            self.dimension.append(dim[i])
        
    def toString(self): 
        s = ''
        for i in self.dimension: s += str(i) + 'x'
        return s[:len(s)-1]


class Tensor(): 

    def __init__(self, data: np.array, shape: list, deltas=None): 
        
        if isinstance(data, np.ndarray): self.elements = data.flatten()                  #float data objects that are stored in sequential memory 
        else: self.elements = np.asarray(data).flatten()
        
        self.shape = shape      # TODO, don't use shape class for now, shape is simply a list or a tuple 
        
        try: 
            np.reshape(self.elements,shape)
        except ValueError:    
            sys.exit("TensorException - cannot represent data {} in given shape {}".format(self.elements,shape))

        if deltas is not None: 
            self.deltas = deltas 
        else: 
            self.deltas = np.zeros(len(self.elements))               #TODO: make this lazy init 
        
        self.loss = 0 
        
    def __str__(self): 
        return '\nTensor with # elements:' + str(len(self.elements)) + ', shape has form:' + str(self.shape) + '\nTensor content: ' + str(np.around(self.elements,2)) + '\nDeltas are: \t' + str(np.around(self.deltas,2))
        
    
    
if __name__ == '__main__': 
    pass