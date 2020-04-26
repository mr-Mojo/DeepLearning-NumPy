from abc import ABC, abstractmethod
import numpy as np 
from tensor_utils import Tensor 
import sys 

# backprop in activation layer = derivative(input) * error of output

DEBUG = False


class InputLayer():                                     #inputlayer transforms list of rawdata into tensors
    def forward(self, rawData: list) -> list: 
        tensors = [] 
        for data in rawData: 
            tensors.append(Tensor(data,np.shape(data)))
        
        return tensors


class AbstractLayer(ABC):     
    @abstractmethod
    def forward(self, inTensors: list, outTensors: list):
        """
            implements the forward pass by filling in outTensors with the processed elements from inTensors
        """
    @abstractmethod 
    def backward(self, outTensors: list, inTensors: list): 
        """
            implements the backward pass by filling in the deltas of the inTensors by processing the elements of the outTensors  
        """
    @abstractmethod
    def param_update(self, inTensors: list, outTensors: list): 
        """ 
            implements the weight update by using the elements of inTensors and outTensors to calculate the delta_weights 
        """


class AbstractActivationLayer(ABC):   
    @abstractmethod
    def forward(self, inTensors: list, outTensors: list):
        pass
        
    @abstractmethod 
    def backward(self, outTensors: list, inTensors: list): 
        pass
    

class SoftmaxLayer(AbstractActivationLayer): 
    
    def __init__(self): 
        self.invalues = None 
    
    #numerically stable version: 
    def softmax(self, x): 
        shiftx = x - np.max(x)
        exps = np.exp(shiftx)
        return exps/np.sum(exps)
    
    def forward(self, inTensors: list, outTensors: list): 
        if self.invalues==None: 
            self.invalues = [t.elements for t in inTensors]

        for i in range(len(inTensors)): 
            x = np.reshape(inTensors[i].elements,inTensors[i].shape)
            outTensors[i].elements = self.softmax(x)
            
    def softmax_derivative(self,Q):
        x=self.softmax(Q).reshape(-1,1)
        return (np.diagflat(x) - np.dot(x, x.T))
        
    # does the same thing as the jacobian
    # outTensors[k].deltas = inTensors[k].deltas @ self.softmax_derivative(self.invalues[k].elements)
    def backward(self, outTensors: list, inTensors: list, update): 
        for k in range(len(self.invalues)):
#   =============================================================================
#             outTensors[k].elements = [t for t in inTensors[k].elements]
#             
#             n = len(inTensors[k].elements)
#             jacobian = np.ones((n,n))*78    #78 dummy to check whether it got overwritten 
#             for i in range(n): 
#                 for j in range(n): 
#                     if i == j: kronecker = 1.0 
#                     else:      kronecker = 0.0 
#                     sm_input = self.invalues[k].elements
#                     sm_output= self.softmax(sm_input)
#                     jacobian[i][j] = outTensors[k].elements[i] * (kronecker - sm_output[j])
#                 
#             outTensors[k].deltas = inTensors[k].deltas @ jacobian
#   =============================================================================
            outTensors[k].deltas = inTensors[k].deltas @ self.softmax_derivative(self.invalues[k]) 
            
    def __str__(self):
        return "Softmax Activation Layer"
            
    
    
    
# ---------------------- < start activation layers > ----------------------


class ReluLayer(AbstractActivationLayer): 
    
    def forward(self, inTensors: list, outTensors: list): 
        self.invalues = []  # [t for t in inTensors]
        for t in inTensors: 
            ten = Tensor(t.elements, t.shape)
            self.invalues.append(ten)
        for i in range(len(inTensors)): 
            for j in range(len(inTensors[i].elements)): 
                if inTensors[i].elements[j] >= 0: 
                    outTensors[i].elements[j] = inTensors[i].elements[j] 
                else: outTensors[i].elements[j] = 0
                
    #d/dx Relu = 0 if x < 0, 1 else 
    def backward(self, outTensors: list, inTensors: list, update): 
        for i in range(len(inTensors)): 
            relu_deriv = np.zeros((len(self.invalues[i].elements),))
            for j in range(len(self.invalues[i].elements)): 
                if self.invalues[i].elements[j] >= 0: 
                    relu_deriv[j] = 1 
            outTensors[i].deltas = relu_deriv * inTensors[i].deltas #deriv(in) * deltas_out 
    
    def __str__(self):
        return "ReLu Activation Layer"


class TanhLayer(AbstractActivationLayer): 
    
    def forward(self, inTensors: list, outTensors: list): 
        self.invalues = [] #[t for t in inTensors]     #store incoming values for backpropagation
        for t in inTensors: 
            ten = Tensor(t.elements,t.shape)
            self.invalues.append(ten)
        for i in range(len(inTensors)): 
            y = np.tanh(inTensors[i].elements)
            outTensors[i] = Tensor(y,np.shape(y))
    
    def backward(self, outTensors: list, inTensors: list, update): 
        for i in range(len(self.invalues)): 
            outTensors[i].elements = np.ones(np.shape(self.invalues[0].elements))  # only helper for dimensions, elements not needed 
            k = np.tanh(self.invalues[i].elements)**2
            
            t1 = 1.0-k
            t1 = np.clip(t1, 1e-1, 0.9)
            #outTensors[i].deltas = (1.0-np.tanh(self.invalues[i].elements)**2)*inTensors[i].deltas
            outTensors[i].deltas = t1*inTensors[i].deltas

    def __str__(self):
        return "Tanh Activation Layer"


class SigmoidLayer(AbstractActivationLayer): 

    def sigm(self,x): 
        x = np.clip(x, -500, 500)        # to avoid under- or overflow
        return 1.0/(1.0+np.exp(-x))
    
    def forward(self, inTensors: list, outTensors: list):
        self.invalues = []
        for t in inTensors: 
            tensor = Tensor(t.elements, t.shape) 
            self.invalues.append(tensor)
        for i in range(len(inTensors)): 
            x = np.reshape(inTensors[i].elements, inTensors[i].shape)
            y = self.sigm(x)
            outTensors[i] = Tensor(y, y.shape)
    
    def backward(self, outTensors: list, inTensors: list, update):      # TODO: SIGM Backprop somewhat buggy
        for i in range(len(inTensors)): 
            outTensors[i].elements = np.ones(np.shape(self.invalues[0].elements))  # only helper for dimensions, elements not needed 

            deriv_input = self.sigm(self.invalues[i].elements)*(1.0-self.sigm(self.invalues[i].elements))
            outTensors[i].deltas = deriv_input*inTensors[i].deltas

            #print(self.invalues[i].elements)
            t1 = (1.0-self.sigm(self.invalues[i].elements)) # this goes to 1, i.e. the 2nd term goes to 0,
            # i.e. self.invalues[i].elements -> great values

            #print(t1)
            #for k in range(len(t1)):
            #    if t1[k]<0.00005: t1[k]=0.00005
            #t2 = self.sigm(self.invalues[i].elements)
            #outTensors[i].deltas = (t1*t2)*inTensors[i].deltas
            
    def __str__(self): 
        return "Sigmoid Activation Layer"
    

# ---------------------- < start fullyConnected layer > ----------------------


class FullyConnectedLayer(AbstractLayer):

    def __init__(self, weights: np.array, bias: np.array, inshape: tuple, outshape: tuple, random_weights=True):
        if random_weights == True:
            #draw weights from a normal distribution with mean 0 and sigma 0.1
            self.weights = np.random.normal(0,0.001,weights.shape[0]*weights.shape[1]).reshape(weights.shape[0],weights.shape[1])

        else:
            self.weights = weights

        self.bias = bias
        self.inshape = inshape
        self.outshape = outshape
        self.delta_weights = np.zeros(weights.shape)
        self.delta_bias = np.zeros(bias.shape)
        self.learning_rate = 0.5
        self.invalues_fw = None

        self.prev_dW = None
        self.prev_error = None

    #y = X*W + b
    def forward(self, inTensors: list, outTensors: list):
        if self.invalues_fw == None:
            self.invalues_fw = [t for t in inTensors]   #copy routine is legitimate
        for i in range(len(inTensors)):
            x = np.reshape(inTensors[i].elements,inTensors[i].shape)
            y = x @ self.weights + self.bias
            outTensors[i] = Tensor(y, y.shape)

    #inTensors already have their deltas set, outTensors still need them 
    def backward(self, outTensors: list, inTensors: list, update, quickProp=False):
        self.invalues_bw = []
        for t in inTensors:     #weird copy routine, but otherwise it will always create a reference datatype???
            tensor = Tensor(t.elements, (np.size(t.elements),1), t.deltas)
            self.invalues_bw.append(tensor)
        for i in range(len(outTensors)):
            outTensors[i].elements = np.ones(np.shape(self.invalues_fw[0].elements))  #only helper for dimensions, elements not needed 
            outTensors[i].deltas = inTensors[i].deltas @ self.weights.transpose()
        if update: self.param_update(quickProp)

    def param_update(self, use_quickProp=False):

        # for quickprop 1st iteration, when t=0 and t-1 can thus not be accessed
        if self.prev_error is None and use_quickProp:
            self.prev_error = [np.zeros(self.invalues_bw[0].deltas.shape)+1e-5 for x in self.invalues_bw]    # prev error per tensor!
            self.prev_dW = [np.random.normal(0, 0.001, np.size(self.weights)).reshape(self.weights.shape) for x in self.invalues_bw]    # prev dW per tensor!
            self.mu = 1.75

        #update weights and bias 
        in_dim = self.weights.shape[0]
        out_dim = self.weights.shape[1]

        for i in range(len(self.invalues_fw)):
            layer_input = self.invalues_fw[i].elements.reshape((1, in_dim))
            error = self.invalues_bw[i].deltas      # dE/dwij

            if use_quickProp:

                error_term = error / (self.prev_error[i]-error) #denominator if sum(denominator) != 0 else error / denominator+1e-5
                deltaW_prev = self.prev_dW[i]

                dLdW = error_term * deltaW_prev
                dLbias = error

                self.prev_dW[i] = dLdW.copy()
                self.prev_error[i] = error.copy()

            else:
                dLdW = layer_input.T @ self.invalues_bw[i].deltas.reshape((1, out_dim))     # input.T @ deltas
                dLbias = self.invalues_bw[i].deltas

            if use_quickProp:
                self.weights = self.weights - dLdW
                self.bias = self.bias - dLbias
            else:
                self.weights = self.weights - self.learning_rate*dLdW
                self.bias = self.bias - self.learning_rate*dLbias
            #self.delta_weights +=  dLdW
            #self.delta_bias += dLbias

    def set_lr(self, new_lr):
        self.learning_rate = new_lr

    def __str__(self):
        return "FullyConnected Layer"
          
# ---------------------- < end fullyConnected layer > ----------------------    
    
    


class Flattening_Layer(AbstractLayer): 
    
    def __init__(self) :
        self.inShapes = [] 
        self.outShapes = [] 
    
    def forward(self, inTensors, outTensors): 
        #inTensors come from a Conv with Shape e.g. (2,3,3) and now need shape (18,)
        
        for i in range(len(inTensors)): 
            self.inShapes.append(inTensors[i].shape)
            newShape = (len(inTensors[i].elements),)
            inTensors[i].shape = newShape 
    
    def backward(self, outTensors, inTensors, update): 
        #inTensors come from a FC-Layer and have shape e.g. (18,) but need (2,3,3)
        
        for i in range(len(outTensors)):
            outTensors[i].shape = self.inShapes[i]
    
    def param_update(self):
        pass
    
# ---------------------- < start Convolution layer > ----------------------
    
    
    
class Convolution2D_Layer(AbstractLayer) :
    
    def __init__(self, input_shape: tuple, kernel: list, weights = None, bias = None, random_weights=True):
        """
            input_shape denotes the original image shape, e.g. (28,28) for MNIST
        """
        
        if not isinstance(input_shape, tuple): 
            sys.exit("INPUT SHAPE MUST BE TUPLE: (depth, x, y)")
        
        if len(kernel) != 4: 
            sys.exit("KERNEL LENGTH MISSMATCH")
        
        if not input_shape[0] == kernel[2]: 
            sys.exit("CANNOT INIT CONV-LAYER WITH DIFF. NO. OF INPUT-DEPTH AND FILTER-DEPTH")
        
        if input_shape[0] < 1: 
            sys.exit('CANNOT HANDLE INPUT_DIM < 1')
            
        if weights is not None: 
            if np.size(weights.shape) < 3:
                sys.exit('NEED WEIGHT SHAPE WITH AT LEAST 3 DIMENSIONS')
        
        self.filtersize_x = kernel[0]
        self.filtersize_y = kernel[1]
        self.filter_depth = kernel[2]       # for MNIST: should always be 1 ? 
        self.filter_count = kernel[3]
        
        #bias is a list: each filter in filter_count has its own bias 
        if bias == None: 
            bias = [] 
            for i in range(self.filter_count): 
                bias.append(0)
        
        self.bias = bias
        self.kernel = kernel 
        self.input_shape = input_shape
        self.learning_rate = 0.1        # later set by optimizer
        
        #random init kernel weights
        #kernel_weights is (filter_count*filter_depth, x,y) np-array
        self.kernel_weights = [] 
        
        
        if weights is None and random_weights is True: 
            
            no_of_elements = self.filtersize_x*self.filtersize_y*self.filter_depth*self.filter_count
            weights = np.random.normal(0,0.001,no_of_elements).reshape((self.filter_depth*self.filter_count, self.filtersize_x, self.filtersize_y))
            self.kernel_weights = weights 
            
            
        elif weights is None and random_weights is False: 
            sys.exit('YOU NEED TO EITHER SPECIFY THE KERNEL WEIGHTS OR SWITCH RANDOM_WEIGHTS FLAG TO TRUE')
        else: 
            self.kernel_weights = weights 
            
            
    #separation of concerns: this simply convolutes what it's given
    #doesnt access or change any layer or kernel values 
    def convolve_2d(self, image, filt, bias):
        kernel_x = filt.shape[1]
        kernel_y = filt.shape[0]
        
        image_y, image_x = image.shape
        outdim_x = image_x - kernel_x + 1
        outdim_y = image_y - kernel_y + 1
        
        if kernel_x > image_x or kernel_y > image_y: 
            sys.exit("KERNEL CANNOT BE BIGGER THAN IMAGE")
        
        convoluted_current = np.zeros((outdim_y, outdim_x)) 
        
        for i in range(outdim_y): 
            for j in range(outdim_x): 
                convoluted_current[i][j] = np.sum(image[i:i+kernel_y, j:j+kernel_x]*filt)
                
        return convoluted_current + bias
    
    
    
    #rotate the kernel in-place
    def rotate_kernel_180(self): 
        for m in range(self.filter_count*self.filter_depth):
            self.kernel_weights[m] = np.rot90(np.rot90(self.kernel_weights[m]))
    
    #flip kernel, so that [x,y,depth,filterCount] becomes [x,y,filterC.,depth]
    def flip_kernel(self): 
        
        if self.input_shape[0] == 1: 
            tmp=self.kernel_weights[1]
            self.kernel_weights[1] = self.kernel_weights[0]
            self.kernel_weights[0] = tmp
        else: 
            tmp = self.kernel_weights[2].copy()
            self.kernel_weights[2] = self.kernel_weights[1]
            self.kernel_weights[1] = tmp    
        
    
    #forward pass was tested and seems robust and correct
    def forward(self, inTensors: list, outTensors: list): 
        self.inshape = inTensors[0].shape   
        self.invalues_fw = [] 
        for t in inTensors: 
            ten = Tensor(t.elements,t.shape)
            self.invalues_fw.append(ten)
        
        for i in range(len(inTensors)): 
            if self.filter_depth != inTensors[i].shape[0]: 
                    sys.exit("ERROR - DEPTH OF INPUT TENSOR MUST BE SAME AS LAYER'S FILTER DEPTH")
            
            if inTensors[i].shape[0] == 1: 
                #if depth is 1 (== just 1 filter): skip depth and reshape to (x,y)
                whole_image = inTensors[i].elements.reshape(inTensors[i].shape[1],inTensors[i].shape[2])
            else:
                whole_image = inTensors[i].elements.reshape(inTensors[i].shape)
                
            count = 0 
            tmp = []
            arrs = []
            
            #get the respective image convoluted with the respective filter 
            #img_ch1 *conv f1_ch1
            #img_ch2 *conv f1_ch2
            #img_ch1 *conv f2_ch1
            #img_ch2 *conv f2_ch2
            for x in range(self.filter_count): 
                for k in range(self.filter_depth):
                    if np.size(whole_image.shape) > 2: 
                        res = self.convolve_2d(whole_image[k],self.kernel_weights[count],self.bias[x]) 
                    else:
                        res = self.convolve_2d(whole_image,self.kernel_weights[count],self.bias[x])
                    tmp.append(res)
                    count += 1 
            
            #sum up the results 
            count = 0
            for x in range(self.filter_depth): 
                arrs.append(tmp[count] + tmp[count+1])
                count += self.filter_depth    
            
            stacked_arrays = np.stack(arrs)
            stacked_arrays = np.clip(stacked_arrays,1e-7,1e7)   #TODO: clip also clips negative values 
            
            outTensor = Tensor(stacked_arrays.flatten(), stacked_arrays.shape)
            outTensor.deltas = np.zeros((len(stacked_arrays.flatten())))
            outTensors[i] = outTensor
        self.outshape = outTensors[0].shape
        
              
                
    def backward(self, outTensors: list, inTensors: list, update): 
        self.outvalues_bw = []
        self.invalues_bw = [] 
        for t in inTensors: 
            ten = Tensor(t.elements, t.shape, t.deltas)
            self.invalues_bw.append(ten)
        
        #transpose kernel so that [x,y,depth,#filters] becomes [x,y,#filters,depth]
        self.flip_kernel()
        
        #rotate kernel
        self.rotate_kernel_180()
        
        #the deltas of the outTensors are the deltas of the inTensor reverse-convoluted with the rotated kernel 
        for i in range(len(inTensors)): 
            
            deltas = inTensors[i].deltas        #some large array, like 12x1 (coming from a 3x4 conv. with [2,2,2,2] kernel)
            delta_shape = (inTensors[i].shape[1],inTensors[i].shape[2])
            

            #map respective outputs to inputs, e.g. 18x1 becomes 2x (2x3) 
            respective_deltas = []          
            curr = 0
            for l in range(1,self.filter_depth+1):
                if self.filter_depth == 1: 
                    respective_deltas.append(deltas.reshape(delta_shape))
                    break
                stopper = (int)(l*len(deltas)/(self.filter_count))
                respective_deltas.append(deltas[curr:stopper].reshape(delta_shape))
                curr = stopper
            
            #pad the deltas to make them ready for convolution that returns inputshaped array: 
            for l in range(len(respective_deltas)):
                px = (int)(np.floor(self.filtersize_x/2))
                py = (int)(np.floor(self.filtersize_y/2))
                respective_deltas[l] = np.pad(respective_deltas[l],(px,py),mode='constant',constant_values=(0,0))
                
                
            # reconvolute: returns #filtercount times  delta-arrays with shape: inshape
            # i.e. here: 2x 4x4 
            
            ## - same routine as in forward -> own method? 
            
            count = 0 
            tmp = []
            arrs = []
            
            #get the respective image convoluted with the respective filter 
            #img_ch1 *conv f1_ch1
            #img_ch2 *conv f1_ch2
            #img_ch1 *conv f2_ch1
            #img_ch2 *conv f2_ch2
            for x in range(self.filter_count): 
                for k in range(self.filter_depth):
                    res = self.convolve_2d(respective_deltas[k],self.kernel_weights[count],self.bias[x]) 
                    tmp.append(res)
                    count += 1 
            
            #sum up the results 
            count = 0
            for x in range(self.filter_depth): 
                arrs.append(tmp[count] + tmp[count+1])
                count += self.filter_count      #TODO: this was 2, is it now really universally applicable? 
            
            
            stacked = np.stack(arrs)
            stacked = np.clip(stacked, 0.1e-5, 1e5)
            
            ten = Tensor(np.ones(len(stacked.flatten())), stacked.shape)         #just put ones to have dummy for dimension 
            ten.deltas = stacked.flatten()
            if DEBUG: print("delta sum: {}".format(np.sum(ten.deltas)))
            
            outTensors[i] = ten
            
            self.outvalues_bw.append(stacked)   #keep the deltas in cache for weight update
            
            # ------------ finished processing all intensors
            
        #redo kernel transpose and rotate so that the layer can be used 
        #for the next forward pass again 
        self.flip_kernel()
        self.rotate_kernel_180()
        
        
        if update: self.param_update()
        
        
    
    
    #for weight update of 1st channel of 1st filter, we need:
    #1st channel of dy and 1st channel of x 
    
    #for weight update of 2nd channel of 1st filter, we need:
    #1st channel of dy and 2nd channel of x 
    
    #in general: 
    # Filter1,Channel1 -> inp[Channel1] * dY[Channel1]
    # Filter1,Channel2 -> inp[Channel2] * dY[Channel1]
    # Filter2,Channel1 -> inp[Channel1] * dY[Channel2]
    # Filter2,Channel1 -> inp[Channel2] * dY[Channel2]
    
    # weight update: dL/df = X *(channelwise_conv) dY
    
    
    def param_update(self):

        for i in range(len(self.invalues_bw)):
            orig_input = self.invalues_fw[i].elements.reshape(self.invalues_fw[i].shape)
            corresp_deltas = self.invalues_bw[i].deltas.reshape(self.invalues_bw[i].shape)
            
            count = 0 
        
            #iterate over delta channels 
            for k in range(corresp_deltas.shape[0]):
                        
                #iterate over input channels 
                for l in range(orig_input.shape[0]): 
                    
                    delta_weights = self.convolve_2d(orig_input[l], corresp_deltas[k], 0)               #TODO: in Markus' sample, the 2nd and 3rd array are switched
                    self.kernel_weights[count] += -self.learning_rate*delta_weights
                    count += 1 
                
                self.bias[k] += -self.learning_rate*np.sum(corresp_deltas[k])
                #self.bias[k] += -self.learning_rate * np.sum(delta_weights)
            if DEBUG: print("bias: {}".format(np.sum(self.bias)))
            
# =============================================================================
#                     filter_update_1_1 = self.convolve_2d(orig_input[0], corresp_deltas[1], 0)
#                     filter_update_1_2 = self.convolve_2d(orig_input[1], corresp_deltas[1], 0)
#                     filter_update_2_1 = self.convolve_2d(orig_input[0], corresp_deltas[2], 0)
#                     filter_update_2_2 = self.convolve_2d(orig_input[1], corresp_deltas[2], 0)
# =============================================================================
                
            
    def __str__(self):
        s = 'Conv Layer Filters:\n\n'
        for x in range(self.filter_count) :
            #s+= 'Filter {}:\n'.format(x)
            for k in range(self.filter_depth): 
                s += '\nFilter {} - Depth {}:\n'.format(x,k)
                s += np.array2string(self.kernel_weights[x][k], precision = 3, separator=',', suppress_small=True)
        
        s += '\n\nwith bias: '
        for element in self.bias: 
            s += str(element)
            s += ','
            
        s += '\n\n\n'
        return s 


    
    
# ---------------------- < start loss layers > ----------------------

class MSE_LossLayer(AbstractActivationLayer): 
    
    def forward(self, inTensors: list, outTensors: list, targetTensors: list):
        for i in range(len(inTensors)): 
            for j in range(len(inTensors[i].elements)):
                outTensors[i].deltas[j] = inTensors[i].elements[j] - targetTensors[i].elements[j]
            outTensors[i].loss = (1.0/(len(outTensors[i].elements))) * sum(outTensors[i].deltas**2)
            #mse loss fkt = 1/N
        return sum([t.loss for t in outTensors])/len(outTensors)
        
    def backward(self, predictedTensors:list, targetTensors: list): 
        pass
        #deltas already filled out in forwardpass 
        #predictedTensors[i].deltas = targetTensors[i].elements - predictedTensors[i].elements
    
    def __str__(self): 
        return "MSE LossLayer"
    
    
class CrossEntropy_LossLayer(AbstractActivationLayer): 
    #cross_ent is defined as -1/N sum(#samples)[ sum(#classes)[t_ij*log(p(i,j))] ]
    #where t_ij is 1 if sample i is in class j, p(i,j) is pred. prob. for sample i to be in class j  
    
    def forward(self, predictedTensors: list, outTensors: list, targetTensors: list):

        res = 0.0 
        N = len(predictedTensors[0].elements)
        for i in range(len(predictedTensors)): 
            res += -np.sum(targetTensors[i].elements*np.log(predictedTensors[i].elements+1e-9))/N
            for j in range(len(predictedTensors[i].elements)):
                outTensors[i].elements[j] = predictedTensors[i].elements[j]
                outTensors[i].deltas[j] = predictedTensors[i].elements[j] - targetTensors[i].elements[j]
            outTensors[i].loss = res  #not really used, just for fun 
        return res 
    
    #backpropagation in cross_entropy: dL/dxi = -target/pred.value (==xi) 
    def backward(self, predictedTensors:list, targetTensors:list): 
        
        for i in range(len(predictedTensors)): 
            predictedTensors[i].deltas = - targetTensors[i].elements/(predictedTensors[i].elements+1e-9)
        
        return predictedTensors
            
    def __str__(self): 
        return "CrossEntropy Losslayer"

# ---------------------- < end loss layers > ----------------------





if __name__ == "__main__": 
    pass
    
# =============================================================================
#     
#     #Beispiel von Markus Folien:     
#
#     in_layer = InputLayer()
#     fc_1 = FullyConnectedLayer(np.array(([-0.5057, 0.3987,-0.8943],
#                                          [ 0.3356, 0.1673, 0.8321],
#                                          [-0.3485,-0.4597,-0.1121])), np.zeros((3,)),(3,),(3,),random_weights=False)
#     sigm = SigmoidLayer()
#     fc_2 = FullyConnectedLayer(np.array(([ 0.4047, 0.9563],
#                                          [-0.8192,-0.1274],
#                                          [ 0.3662,-0.7252])),np.zeros((2,)),(3,),(2,),random_weights=False)
#     out  = SoftmaxLayer() 
#     loss = CrossEntropy_LossLayer()
#     
#     inputTensor  = [Tensor(np.array(([0.4183, 0.5209, 0.0291])), (3,))]
#     targetTensor = [Tensor(np.array(([0.7095,0.0942])), (2,))]
#     
#     fc_1.forward(inputTensor,inputTensor)
#     sigm.forward(inputTensor,inputTensor)
#     fc_2.forward(inputTensor,inputTensor)
#     out.forward(inputTensor,inputTensor)
#     loss.forward(inputTensor,inputTensor,targetTensor)
#     
#     predictedTensors = loss.backward(inputTensor,targetTensor)
#     
#     out.backward(predictedTensors, predictedTensors)
# =============================================================================
    
