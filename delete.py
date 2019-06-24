

import numpy as np 
import layer_utils
import tensor_utils 
import network_utils
from layer_utils import Convolution2D_Layer, FullyConnectedLayer, SoftmaxLayer, CrossEntropy_LossLayer, Flattening_Layer
from tensor_utils import Tensor
from network_utils import NeuronalNetwork
import matplotlib.pyplot as plt 

import tensorflow as tf 
mnist = tf.keras.datasets.mnist
(x_train, y_train),(x_test, y_test) = mnist.load_data()     #np.ndarray, shape (28,28)
x_train, x_test = x_train / 255.0, x_test / 255.0           #values normalized [0,1] 


## --------------------------------------- init layers -----------------------------------------------------

### ----------- NETWORK STRUCTURE 

# conv2d                #shape (1,28,28) -> (1,24,24)
# conv2d                #shape (1,24,24) -> (1,20,20)
# Flattening            #shape (1,20,20) -> (400,)
# FullyConnected        #shape (400,)    -> (10,)
# Softmax               #shape (10,)     -> (10,)
# CrossEntropy          #shape (10,)     -> (10,) probab.

# =============================================================================
# conv = Convolution2D_Layer((2,28,28),[5,5,2,2],random_weights=True)
# flat = Flattening_Layer()
# full = FullyConnectedLayer(weights=np.ones((1152,10)),bias=np.zeros((10,)),inshape=(1152,),outshape=(10,))
# soft = SoftmaxLayer()
# loss = CrossEntropy_LossLayer()
# =============================================================================
 
conv = Convolution2D_Layer((1,28,28),[5,5,1,2],random_weights=True)
flat = Flattening_Layer()
full = FullyConnectedLayer(weights=np.ones((576,10)),bias=np.zeros((10,)),inshape=(576,),outshape=(10,))
soft = SoftmaxLayer()
loss = CrossEntropy_LossLayer()

#layers = [conv, conv2, flat, full, soft, loss]
layers = [conv, flat, full, soft, loss]

## --------------------------------------- init network -----------------------------------------------------

nn = network_utils.NeuronalNetwork(0, layers, [], [])
trainer = network_utils.SGDTrainer()

## --------------------------------------- init in-/output -----------------------------------------------------

inputList = []  
testInput = []
testTarget= []


for t in x_train[2:20]: 
    img = t.reshape(1,28,28)
    #img = np.stack((img,img),axis=0)
    ten = layer_utils.Tensor(img,img.shape)
    inputList.append(ten)

targetList = [] 
for t in y_train[2:20]: 
    tar = np.zeros((10,1))
    tar[t] = 1 
    ten = layer_utils.Tensor(tar.flatten(), (10,))
    targetList.append(ten)

for t in x_test: 
    img = t.reshape(1,28,28)
    #img = np.stack((img,img),axis=0)
    ten = layer_utils.Tensor(img,img.shape)
    testInput.append(ten)

for t in y_test: 
    tar = np.zeros((10,1))
    tar[t] = 1 
    ten = layer_utils.Tensor(tar.flatten(), (10,))
    testTarget.append(ten)



nn = trainer.optimize(nn,inputList,targetList,epochs=100, lr=0.1)



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
# =============================================================================
#     
# if test_network: 
#     count = 0 
#     res = nn.forward(testInput, testTarget)
#     for i in range(len(testInput)): 
#         if np.argmax(res[i].elements) == np.argmax(testTarget[i]):
#             count += 1 
#     
#     print("\n ------------------------------------------- \n\nAccuracy on the test set: {0:.2f}%".format(count*100/len(testInput)))
# 
# =============================================================================


# =============================================================================
# 
# ### ----------- NETWORK STRUCTURE 
# 
# # conv2d                #shape (2,3,4) -> (2,3,2)
# # Flattening            #shape (2,3,2) -> (12,)
# # FullyConnected        #shape (12,)   -> (10,)
# # Softmax               #shape (10,)   -> (10,)
# # CrossEntropy          #shape (10,)   -> (10,) probab.
# 
# conv = Convolution2D_Layer((2,28,28),[2,2,2,2],random_weights=True)
# flat = Flattening_Layer()
# full = FullyConnectedLayer(weights=np.ones((1458,10)),bias=np.zeros((10,)),inshape=(1458,),outshape=(10,))
# soft = SoftmaxLayer()
# loss = CrossEntropy_LossLayer()
# 
# layers = [conv,flat,full,soft,loss]
# 
# 
# for i in range(10):
#     for l in layers[:-1]: 
#         l.forward(outList,outList)
#     loss.forward(outList,outList,tarList)
#     res = loss.backward(outList,tarList)
#     total_loss = sum([t.loss for t in res])
#     print(total_loss)
#     
#     for l in range(len(layers)-2,-1,-1): 
#         layer = layers[l]
#         layer.backward(outList,outList)
#     
# 
# 
# conv.forward(inList, outList)
# flat.forward(outList,outList)
# full.forward(outList,outList)
# soft.forward(outList,outList)
# loss.forward(outList,outList,tarList)
# 
# print(outList[0])
# print(outList[0].loss)
# 
# loss.backward(outList,tarList)
# soft.backward(outList,outList)
# full.backward(outList,outList)
# flat.backward(outList,outList)
# conv.backward(outList,outList)
# 
# conv.forward(inList, outList)
# flat.forward(outList,outList)
# full.forward(outList,outList)
# soft.forward(outList,outList)
# loss.forward(outList,outList,tarList)
# 
# print(outList[0])
# print(outList[0].loss)
# 
# loss.backward(outList,tarList)
# soft.backward(outList,outList)
# full.backward(outList,outList)
# flat.backward(outList,outList)
# conv.backward(outList,outList)
# 
# conv.forward(inList, outList)
# flat.forward(outList,outList)
# full.forward(outList,outList)
# soft.forward(outList,outList)
# loss.forward(outList,outList,tarList)
# 
# print(outList[0])
# print(outList[0].loss)
# 
# =============================================================================




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
