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
 
    conv = Convolution2D_Layer((1,28,28),[10,10,1,2],random_weights=True)
    flat = Flattening_Layer()
    full = FullyConnectedLayer(np.ones((361,10)),np.zeros((10,)),(361,),(10,),random_weights=True)
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

    n = 500

    for t in x_train[0:n]: 
        img = t.reshape(1,28,28)
        ten = layer_utils.Tensor(img,img.shape)
        inputList.append(ten)
    
    targetList = [] 
    for t in y_train[0:n]: 
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
    
    nn = trainer.optimize(nn,inputList,targetList,epochs=50, lr=0.1)


    ## --------------------------------------- annotations -----------------------------------------------------
    
    
    #good param choice: 
    # n = 50, lr = 0.1, e = 50, kernel = [10,10,1,2]
    # n = 500 same, but quite slow 
    
    #very sensitive to learning_rate changes
    #0.05 is a good value for n = 20, 0.001 for n = 50 but takes >30 epochs 

    #kernel setting (10,10,1,2) lr 0.01 fc 361
    
    
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
                #print(res[j])
                plt.imshow(np.reshape(inputList[j].elements,(28,28)))
                plt.show()
                
    if test_network: 
        count = 0 
        stopper = 200 
        res_test = nn.forward(testInput[0:stopper], testTarget[0:stopper])
        for i in range(len(res_test)): 
            if np.argmax(res_test[i].elements) == np.argmax(testTarget[i].elements):
                count += 1 
        
        print("\n ------------------------------------------- \n\nAccuracy on the test set: {0:.2f}%".format(count*100/len(testInput[0:stopper])))
         


