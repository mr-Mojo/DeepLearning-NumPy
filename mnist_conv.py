import numpy as np 
import layer_utils
import network_utils

from layer_utils import Convolution2D_Layer, FullyConnectedLayer, SoftmaxLayer, CrossEntropy_LossLayer, Flattening_Layer
from tensor_utils import Tensor
from network_utils import NeuronalNetwork

import matplotlib.pyplot as plt 


if __name__ == "__main__": 
    
    ## --------------------------------------- init mnist -----------------------------------------------------
    
    import tensorflow as tf 
    mnist = tf.keras.datasets.mnist
    (x_train, y_train),(x_test, y_test) = mnist.load_data()     #np.ndarray, shape (28,28)
    x_train, x_test = x_train / 255.0, x_test / 255.0           #values normalized [0,1] 


    ## --------------------------------------- init layers -----------------------------------------------------
    ### ----------- NETWORK STRUCTURE 

    #convolution    (1,28,28) *conv (5,5,1,2) -> (1,24,24)
    #flattening     (1,24,24)   -> (576,)
    #fully_conn.    (576,)      -> (576,10)
    #softmax_act    (576,10)    -> (10,)
    #crossent_loss  (10,)       -> (10,) probability 
 
    conv = Convolution2D_Layer((1,28,28),[5,5,1,2],random_weights=True)
    flat = Flattening_Layer()
    full = FullyConnectedLayer(np.ones((576,10)),np.zeros((10,)),(576,),(10,),random_weights=True)
    soft = SoftmaxLayer()
    loss = CrossEntropy_LossLayer()

    layers = [conv, flat, full, soft, loss]

    ## --------------------------------------- init network -----------------------------------------------------

    nn = network_utils.NeuronalNetwork(0, layers, [], [])
    trainer = network_utils.SGDTrainer()


    ## --------------------------------------- init in-/output -----------------------------------------------------

    inputList = []  
    testInput = []
    testTarget= []


    for t in x_train[2:50]: 
        img = t.reshape(1,28,28)
        ten = layer_utils.Tensor(img,img.shape)
        inputList.append(ten)
    
    targetList = [] 
    for t in y_train[2:50]: 
        tar = np.zeros((10,1))
        tar[t] = 1 
        ten = layer_utils.Tensor(tar.flatten(), (10,))
        targetList.append(ten)
    
    for t in x_test: 
        img = t.reshape(1,28,28)
        ten = layer_utils.Tensor(img,img.shape)
        testInput.append(ten)
    
    for t in y_test: 
        tar = np.zeros((10,1))
        tar[t] = 1 
        ten = layer_utils.Tensor(tar.flatten(), (10,))
        testTarget.append(ten)
    
    
    
    
    ## --------------------------------------- train the nn -----------------------------------------------------

    #very sensitive to learning_rate changes, 0.05 is a good value for n = 20, 0.001 for n = 50 but takes >30 epochs 
    
    nn = trainer.optimize(nn,inputList,targetList,epochs=100, lr=0.001)


    
    ## --------------------------------------- show the results -----------------------------------------------------


    show_results = True
    test_network = True
       
    
    if show_results: 
            res = nn.forward(inputList, targetList)
            start_index = np.random.randint(0,len(res))
            for i in range(len(res[2:8])): 
                j = np.random.randint(0,len(res))
                mnist_image = inputList[j]
                mnist_label = np.argmax(targetList[j].elements)
                prediction = np.argmax(res[j].elements)
                
                print("\nprediction for the following image: {}\nlabel for the following image: {}".format(prediction, mnist_label))
                print(res[j])
                plt.imshow(np.reshape(inputList[j].elements,(28,28)))
                plt.show()
                
    if test_network: 
        count = 0 
        stopper = 200 
        res = nn.forward(testInput[0:stopper], testTarget[0:stopper])
        print(res[0])
        print(testTarget[0])
        for i in range(len(res)): 
            if np.argmax(res[i].elements) == np.argmax(testTarget[i]):
                count += 1 
        
        print("\n ------------------------------------------- \n\nAccuracy on the test set: {0:.2f}%".format(count*100/len(testInput[0:stopper])))
         
















# =============================================================================
# Markus' example, works fine. 
#
# w1 = np.array(([[0.1,-0.2],[0.3,0.4]],
#                [[0.7,0.6],[0.9,-1.1]],
#                [[0.37,-0.9],[0.32,0.17]],
#                [[0.9,0.3],[0.2,-0.7]]))
# 
# conv = Convolution2D_Layer((2,3,4),[2,2,2,2],w1,random_weights=False)
# 
# elements = np.array([[[0.1,-0.2,0.5,0.6],
#                       [1.2,1.4,1.6,2.2],
#                       [0.01,0.2,-0.3,4.0]],
# 
#                         [[0.9,0.3,0.5,0.65],
#                          [1.1,0.7,2.2,4.4],
#                          [3.2,1.7,6.3,8.2]]])
# 
# ten = Tensor(elements,elements.shape)
# 
# inList = [ten]
# outList = [ten]
# 
# conv.forward(inList,outList)
# outList[0].deltas = np.array([[[0.1,0.33,-0.6],
#                               [-0.25,1.3,0.01]],
# 
#                             [[-0.5,0.2,0.1],
#                              [-0.8,0.81,1.1]]]).flatten()
# 
# conv.backward(outList,outList)
# print(outList[0])
# =============================================================================




#small example to verify that the network is working correclty --> IT DOES! THE LOSS DECREASES !!! 

### ----------- INPUT & TARGET DATA 
# =============================================================================
# #works with 3x4
# elements = np.array([[[0.1,-0.2,0.5,0.6],
#                       [1.2,1.4,1.6,2.2],
#                       [0.01,0.2,-0.3,4.0]],
# 
#                         [[0.9,0.3,0.5,0.65],
#                          [1.1,0.7,2.2,4.4],
#                          [3.2,1.7,6.3,8.2]]])
# 
# =============================================================================

# =============================================================================
# # now trying with 3x3 
# elements = np.array([[[0.1,-0.2,0.5],
#                       [1.2,1.4,1.6],
#                       [0.01,0.2,-0.3]],
# 
#                         [[0.9,0.3,0.5],
#                          [1.1,0.7,2.2],
#                          [3.2,1.7,6.3]]])
# 
# =============================================================================

