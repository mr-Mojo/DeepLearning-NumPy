# Deep Learning
A small deep learning framework, built from scratch using Python and NumPy. It it readily usable for MNIST classification with FC layers (`mnist_fc`) and convolutional layers (`mnist_conv`) and can be extended arbitrarily. 

### About
The framework uses the `tensor` class to hold, propagate and process information. Each tensor holds its own `elements` and has a `shape` parameter. The `NeuralNetwork` class can be used to instantiate a network with a list of its layers. The network will use Stochastic Gradient Descent `SGD` and Backpropagation to optimize its weights. It can be extended with arbitrary optimizers and optimization algorithms.  

### Layers 
The framework provides the following layers: 
* **Pre-Processing:** 
  * Input
  * Flattening 
* **Activation Functions:**
  * Sigmoid 
  * Tanh
  * ReLU
  * Softmax
* **Processing Layers:**
  * Fully-Connected (FC)
  * Convolution2D 
* **Loss Layers:**
  * MSE 
  * Crossentropy

