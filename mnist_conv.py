import numpy as np 
import matplotlib.pyplot as plt 
import pickle


import layer_utils
import tensor_utils 
import network_utils



if __name__ == '__main__': 

    ## --------------------------------------- init mnist -----------------------------------------------------
    
    import tensorflow as tf 
    mnist = tf.keras.datasets.mnist
    (x_train, y_train),(x_test, y_test) = mnist.load_data()     #np.ndarray, shape (28,28)
    x_train, x_test = x_train / 255.0, x_test / 255.0           #values normalized [0,1] 
    
    
    ## --------------------------------------- init layers -----------------------------------------------------
    
    in_layer = layer_utils.InputLayer()
    
    conv1 = layer_utils.Convolution2D_Layer((28,28), np.eye(3,3))
    conv2 = layer_utils.Convolution2D_Layer((28,28), np.eye(3,3))
    
    architecture = [conv1, conv2]
    
    
    layers = architecture
    
    ## --------------------------------------- init network -----------------------------------------------------
    
    nn = network_utils.NeuronalNetwork(in_layer, layers, [], [])
    trainer = network_utils.SGDTrainer()
    
    ## --------------------------------------- init in-/output -----------------------------------------------------
    
    inputList = []  
    
    for t in x_train[2:20]: 
        inputList.append(t.flatten())

    targetList = [] 
    for t in y_train[2:20]: 
        tar = np.zeros((10,1))
        tar[t] = 1 
        targetList.append(tar.flatten())

    ## --------------------------------------- train the nn -----------------------------------------------------
    
# =============================================================================
#     nn = trainer.optimize(nn,inputList,targetList,epochs=500, lr=0.5)
#     res = nn.forward(inputList, targetList)
#     
#     show_results = True
#     
#     if show_results: 
#         for i in range(len(res[0:5])): 
#             j = np.random.randint(0,len(res))
#             mnist_image = inputList[j]
#             mnist_label = np.argmax(targetList[j])
#             prediction = np.argmax(res[j].elements)
#             
#             print("\nprediction for the following image: {}\nlabel for the following image: {}".format(prediction, mnist_label))
#             print(res[j])
#             plt.imshow(np.reshape(inputList[j],(28,28)))
#             plt.show()
#         
# =============================================================================
        
    
# =============================================================================
#     
#     #compare against tensorflow to validate feasibility of architecture
#     import tensorflow as tf 
#     mnist = tf.keras.datasets.mnist
#     
#     (x_train, y_train),(x_test, y_test) = mnist.load_data()
#     x_train, x_test = x_train / 255.0, x_test / 255.0
#     
#     x_train = x_train.reshape(-1,28,28,1)
#     x_test = x_test.reshape(-1,28,28,1)
#     
#     model = tf.keras.models.Sequential([
#       tf.keras.layers.Conv2D(1, kernel_size=(5,5), strides=(1,1), input_shape=(28,28,1), activation=tf.nn.relu),
#       tf.keras.layers.Flatten(),
#       tf.keras.layers.Dense(512, activation=tf.nn.sigmoid),
#       tf.keras.layers.Dense(10, activation=tf.nn.softmax)
#     ])
#     model.compile(optimizer='adam',
#                   loss='sparse_categorical_crossentropy',
#                   metrics=['accuracy'])
#     
#     model.fit(x_train, y_train, epochs=5)
#     model.evaluate(x_test, y_test)
# 
# =============================================================================
