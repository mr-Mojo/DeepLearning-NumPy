import tensor_utils
import layer_utils 
from typing import List 
import numpy as np 

class NeuronalNetwork(): 
    def __init__(self, inputLayer: layer_utils.InputLayer, layerList: List, 
                 parameterList: List, deltaparameterList: List): 
        
        if isinstance(inputLayer, layer_utils.InputLayer):
            self.inputLayer = inputLayer 
            self.hasInputLayer = True 
        else: 
            self.hasInputLayer = False
        
        self.layers =       layerList               #list<Layer> 
        self.parameters =   parameterList           #list<weights,biases>
        self.deltaParams =  deltaparameterList      #list<deltaweights,deltabiases>
        
        self.in_tensors = None
        self.out_tensors = None
    
    def forward(self, data:list, targets:layer_utils.Tensor):
        if not isinstance(targets[0], layer_utils.Tensor): 
            targets = self.inputLayer.forward(targets)
        if not isinstance(data[0], np.ndarray) and not isinstance(data[0], layer_utils.Tensor): 
            print("ERROR - only numpyArrays can be handled by input layer")
            return
        
        
        #input layer: 
        if self.hasInputLayer: inTensors = self.inputLayer.forward(data)
        else: inTensors = data.copy()
        
        #all layers but the last: 
        for layer in self.layers[:-1]: layer.forward(inTensors, inTensors)
        
        
        loss_layer = self.layers[-1] 
        outTensors = inTensors.copy()
        
        #loss_layer 
        loss_layer.forward(inTensors, outTensors, targets)
        
        return outTensors  

    
    
    def backprop(self,predictedTensors,targetTensors): 
        
        loss_layer = self.layers[-1] 
        loss_layer.backward(predictedTensors, targetTensors)
        
        for i in range(len(self.layers)-2,-1,-1):   #from second last list element to 0th element 
            self.layers[i].backward(predictedTensors, predictedTensors)
        
        return predictedTensors         #return useless, just debug 
    
    def __str__(self): 
        out = '' 
        for l in self.layers: 
            if isinstance(l,layer_utils.FullyConnectedLayer):
                weights = l.weights.flatten()
                for w in weights: 
                    out += str(w)
            out += '\n'
        return out
    
    

class SGDTrainer(): 
    def __init__(self, batchsize=3, shuffle=True, update_mechanism='vanillaGD'): 
        self.batchsize = batchsize
        self.shuffle = shuffle
        self.update_mechanism = update_mechanism    # currently only vanilla GD is implemented 
        
        
    def optimize(self, network : NeuronalNetwork, data: list, labels: list, epochs=50, lr=0.5, early_stopping=True):
        self.learning_rate = lr 
        early_stopping_treshold = 0.00000005
        
        #github test
        # ------------------ sanity checks ------------------------------
        if not isinstance(labels[0], tensor_utils.Tensor): 
            targetTensors = network.inputLayer.forward(labels)
        else: targetTensors = labels
        
        if not len(data)==len(labels): print("ERROR - DATA AND LABELS DIFFER IN SIZE")
        # ----------------- /sanity checks ------------------------------
        
        for l in network.layers: 
            if isinstance(l, layer_utils.FullyConnectedLayer) or isinstance(l, layer_utils.Convolution2D_Layer): 
                l.learning_rate = self.learning_rate
        
        isOne = True
        isTwo = False
        for i in range(epochs): 
            
            
            res = network.forward(data,targetTensors)
            loss = sum([t.loss for t in res])/len(res)
            
            
            acc = 0
            for t in range(len(res)): 
                if np.argmax(res[t].elements) == np.argmax(targetTensors[t].elements):
                    if res[t].elements[np.argmax(res[t].elements)] > 0.8: acc+=1
        
            
            print("mean loss at epoch {0}: {1:.5f} -- Acc: {2:.2f}%".format(i+1,loss,acc*100/len(res)))
            
            if i % 21 == 0 and isOne and i > 20: 
                isOne = False
                isTwo = True
                curr_loss = loss 
            
            if i % 40 == 0 and isTwo: 
                isOne = True
                isTwo = False
                if np.around(curr_loss,3) == np.around(loss,3): 
                    print("BOOSTING LEARNING RATE")
                    self.learning_rate *= 2
                    for l in network.layers: 
                        if isinstance(l, layer_utils.FullyConnectedLayer) or isinstance(l, layer_utils.Convolution2D_Layer):
                            l.learning_rate = self.learning_rate
            
            network.backprop(res,targetTensors)
            
           
            if early_stopping: 
                if loss < early_stopping_treshold:
                    return network
         
            

        return network  #return ununsed, just for debug 

    