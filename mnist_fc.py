import numpy as np 
import matplotlib.pyplot as plt 


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

    # not working yet, probably gradient problems somewhere
    fc1 = layer_utils.FullyConnectedLayer(np.zeros((784,64)), np.zeros((64,)), (784,), (64,), random_weights=True)
    act1= layer_utils.SigmoidLayer()
    fc2 = layer_utils.FullyConnectedLayer(np.zeros((64,32)), np.zeros((32,)), (64,), (32,), random_weights=True)
    act2= layer_utils.SigmoidLayer()
    fc3 = layer_utils.FullyConnectedLayer(np.zeros((32,10)), np.zeros((10,)), (32,), (10,), random_weights=True)
    out1= layer_utils.SoftmaxLayer()
    loss= layer_utils.CrossEntropy_LossLayer()
    architecture1 = [fc1, act1, fc2, act2, fc3, out1, loss]

    # works with RELU and Tanh
    # re-do sigm backprop
    dense1 = layer_utils.FullyConnectedLayer(np.zeros((784, 32)), np.zeros((32,)), (784,), (32,), random_weights=True)
    activ1_t = layer_utils.TanhLayer()
    activ_1_relu = layer_utils.ReluLayer()
    dense2 = layer_utils.FullyConnectedLayer(np.zeros((32, 10)), np.zeros((10,)), (32,), (10,), random_weights=True)
    dense3 = layer_utils.SoftmaxLayer()
    lossfc = layer_utils.CrossEntropy_LossLayer()
    lossmse = layer_utils.MSE_LossLayer()
    architecture_crossentr = [dense1, activ_1_relu, dense2, dense3, lossfc]
    architecture_mse = [dense1, activ1_t, dense2, dense3, lossmse]

    a3_dense1 = layer_utils.FullyConnectedLayer(np.zeros((784, 10)), np.zeros((10,)), (784,), (10,), random_weights=True)
    a3_act = layer_utils.SoftmaxLayer()
    a3_loss = layer_utils.CrossEntropy_LossLayer()
    architecture3 = [a3_dense1, a3_act, a3_loss]

    layers = architecture_mse

    # also works really well with Tanh and MSE, lr = 0.1
    
    ## --------------------------------------- init network -----------------------------------------------------
    
    nn = network_utils.NeuronalNetwork(in_layer, layers, [], [])
    trainer = network_utils.SGDTrainer()
    
    ## --------------------------------------- init in-/output -----------------------------------------------------
    
    inputList = []  
    testInput = []
    testTarget= []
    
    n = 500
    
    for t in x_train[0:n]: 
        inputList.append(t.flatten())

    targetList = [] 
    for t in y_train[0:n]: 
        tar = np.zeros((10,1))
        tar[t] = 1 
        targetList.append(tar.flatten())

    for t in x_test: 
        testInput.append(t.flatten())
    
    for t in y_test: 
        tar = np.zeros((10,1))
        tar[t] = 1 
        testTarget.append(tar.flatten())

    ## --------------------------------------- train the nn -----------------------------------------------------

    nn = trainer.optimize(nn,inputList,targetList,epochs=300, lr=0.08, use_quickProp=False)

    ## --------------------------------------- annotations -----------------------------------------------------
    
    #25.03.2020, 09:30 Uhr: 
    #avg. time for n = 20000: 4.2  s  
    #avg. time for n = 60000: 17.5 s

    #when using arch3, lr=0.1 for all n 
    #when using relu with n = 2000 use lr = 0.005
    #when using with tanh n = 200 use lr=0.01 epochs 2000 
    #when using with sigm n = 200 use lr=0.01 epochs 2000 
    
    #when using with mse and tanh e=1000, lr=0.1, n=100 
    
    #TODO for speedup: copy routine in forward layers 
	
    ## --------------------------------------- eval/plot results -----------------------------------------------------
    
    show_results = True
    test_network = True
    
    if show_results: 
        res = nn.forward(inputList, targetList)
        start_index = np.random.randint(0,len(res)-10)
        for i in range(len(res[start_index:start_index+10])): 
            j = np.random.randint(0,len(res))
            mnist_image = inputList[j]
            mnist_label = np.argmax(targetList[j])
            prediction = np.argmax(res[j].elements)
            
            print("\nprediction for the following image: {}\nlabel for the following image: {}".format(prediction, mnist_label))
            #print(res[j])
            plt.imshow(np.reshape(inputList[j],(28,28)))
            plt.show()
    
    if test_network: 
        count = 0 
        res = nn.forward(testInput, testTarget)
        for i in range(len(testInput)): 
            if np.argmax(res[i].elements) == np.argmax(testTarget[i]):
                count += 1 
        
        print("\n ------------------------------------------- \n\nAccuracy on the test set: {0:.2f}%".format(count*100/len(testInput)))
            
# =============================================================================
#     
#     #compare against tensorflow to validate feasibility of architecture
#     import tensorflow as tf 
#     mnist = tf.keras.datasets.mnist
#     
#     (x_train, y_train),(x_test, y_test) = mnist.load_data()
#     x_train, x_test = x_train / 255.0, x_test / 255.0
#     
#     model = tf.keras.models.Sequential([
#       tf.keras.layers.Flatten(input_shape=(28, 28)),
#       tf.keras.layers.Dense(32, activation=tf.nn.relu),
#       tf.keras.layers.Dense(10, activation=tf.nn.softmax)
#     ])
#     model.compile(optimizer='sgd',
#                   loss='sparse_categorical_crossentropy',
#                   metrics=['accuracy'])
#     
#     model.fit(x_train, y_train, epochs=50)
#     model.evaluate(x_test, y_test)
# =============================================================================


        