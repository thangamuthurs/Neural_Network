#!/usr/bin/env python
# coding: utf-8

# ### Deep L layered Neural Network
# 
# * Here, I am constructing a Deep Neural Network of L layers to identify the image shared by user is Cat or Dog
# * I used the images from Kaggle for training and testing purposes

# #### Libraries Required

# In[1]:


import numpy as np
import h5py
import matplotlib.pyplot as plt
import numpy as np
import time
import scipy
from PIL import Image
from scipy import ndimage
import copy
get_ipython().run_line_magic('matplotlib', 'inline')
plt.rcParams['figure.figsize'] = (5.0, 4.0) # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')

np.random.seed(1)


# #### Parameters Initialization

# In[2]:


# GRADED FUNCTION: initialize_parameters

def initialize_parameters(n_x, n_h, n_y):
    np.random.seed(1)

    W1 = np.random.randn(n_h,n_x) * 0.01
    b1 = np.zeros((n_h,1))
    W2 = np.random.randn(n_y,n_h) * 0.01
    b2 = np.zeros((n_y,1))

    parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2}

    return parameters


# In[3]:


parameters = initialize_parameters(3,2,1)

print("W1 = " + str(parameters["W1"]))
print("b1 = " + str(parameters["b1"]))
print("W2 = " + str(parameters["W2"]))
print("b2 = " + str(parameters["b2"]))


# In[4]:


# GRADED FUNCTION: initialize_parameters_deep

def initialize_parameters_deep(layer_dims):
    np.random.seed(3)
    parameters = {}
    L = len(layer_dims) # number of layers in the network

    for l in range(1, L):
        parameters['W'+str(l)] = np.random.randn(layer_dims[l],layer_dims[l-1]) * 0.01
        parameters['b'+str(l)] = np.zeros((layer_dims[l],1))

        assert(parameters['W' + str(l)].shape == (layer_dims[l], layer_dims[l - 1]))
        assert(parameters['b' + str(l)].shape == (layer_dims[l], 1))


    return parameters


# In[10]:


print("Test Case 1:\n")
parameters = initialize_parameters_deep([5,4,3])

print("W1 = " + str(parameters["W1"]))
print("b1 = " + str(parameters["b1"]))
print("W2 = " + str(parameters["W2"]))
print("b2 = " + str(parameters["b2"]))


# #### Sigmoid & ReLU function 

# In[11]:


def sigmoid(Z):
    A = 1 / (1 + np.exp(-Z))
    cache = Z
    return A, cache

def relu(Z):
    A = np.maximum(0, Z)
    cache = Z
    return A, cache


# ##### Linear forward

# In[12]:


def linear_forward(A, W, b):
    Z = np.dot(W,A)+b
    cache = (A, W, b)
    return Z, cache


# In[13]:


#Test Case:
t_A, t_W, t_b = [[ 1.62434536, -0.61175641],
        [-0.52817175, -1.07296862],
        [ 0.86540763, -2.3015387 ]], [[ 1.74481176, -0.7612069 ,  0.3190391 ]], [[-0.24937038]]
t_Z, t_linear_cache = linear_forward(t_A, t_W, t_b)
print("Z = " + str(t_Z))


# ##### Activation function, linear part in forward propogation

# In[14]:


def linear_activation_forward(A_prev, W, b, activation):
    if activation == "sigmoid":
        Z, linear_cache = linear_forward(A_prev, W, b)
        A, activation_cache = sigmoid(Z)

    elif activation == "relu":
        Z, linear_cache = linear_forward(A_prev, W, b)
        A, activation_cache = relu(Z)

    cache = (linear_cache, activation_cache)

    return A, cache


# In[15]:


# Test cases
t_A_prev, t_W, t_b = [[-0.41675785, -0.05626683],
        [-2.1361961 ,  1.64027081],
        [-1.79343559, -0.84174737]], [[ 0.50288142, -1.24528809, -1.05795222]], [[-0.90900761]]

t_A, t_linear_activation_cache = linear_activation_forward(t_A_prev, t_W, t_b, activation = "sigmoid")
print("With sigmoid: A = " + str(t_A))

t_A, t_linear_activation_cache = linear_activation_forward(t_A_prev, t_W, t_b, activation = "relu")
print("With ReLU: A = " + str(t_A))


# #### Forward propogation: Linear -> ReLU for L-1 layers and Sigmoid function for final layer

# In[16]:


def L_model_forward(X, parameters):
    caches = []
    A = X
    L = len(parameters) // 2                  # number of layers in the neural network
    for l in range(1, L):
        A_prev = A
        A, cache = linear_activation_forward(A_prev,  parameters['W'+str(l)],  parameters['b'+str(l)], "relu")
        caches.append(cache)
    
    AL,cache = linear_activation_forward(A,  parameters['W'+str(L)],  parameters['b'+str(L)], "sigmoid")
    caches.append(cache)
    
    return AL, caches


# ##### Computing Cost

# In[17]:


# GRADED FUNCTION: compute_cost

def compute_cost(AL, Y):
    m = Y.shape[1]
    multiply1 = np.dot(Y,np.log(AL).T)
    multiply2 = np.dot((1-Y),np.log(1-AL).T)
    cost = -(1/m)*np.sum(multiply1+multiply2)

    cost = np.squeeze(cost)


    return cost


# In[18]:


t_Y, t_AL = np.array([[1, 1, 0]]), np.array([[0.8, 0.9, 0.4]])
t_cost = compute_cost(t_AL, t_Y)

print("Cost: " + str(t_cost))


# ##### Linear part in activation function on backward propagation

# In[19]:


def linear_backward(dZ, cache):
    A_prev, W, b = cache
    m = A_prev.shape[1]
    dW = (1/m)*np.dot(dZ, A_prev.T)
    db = (1/m)*np.sum(dZ, axis = 1, keepdims = True)
    dA_prev = np.dot(W.T, dZ)
    
    return dA_prev, dW, db


# In[20]:


t_dZ= np.array([[ 1.62434536, -0.61175641, -0.52817175, -1.07296862],
                [ 0.86540763, -2.3015387 ,  1.74481176, -0.7612069 ],
                [ 0.3190391 , -0.24937038,  1.46210794, -2.06014071]]),
t_linear_cache = np.array([[-0.3224172 , -0.38405435,  1.13376944, -1.09989127],
                          [-0.17242821, -0.87785842,  0.04221375,  0.58281521],
                          [-1.10061918,  1.14472371,  0.90159072,  0.50249434],
                          [ 0.90085595, -0.68372786, -0.12289023, -0.93576943],
                         [-0.26788808,  0.53035547, -0.69166075, -0.39675353]]), np.array([[-0.6871727 , -0.84520564, -0.67124613, -0.0126646 , -1.11731035],
                          [ 0.2344157 ,  1.65980218,  0.74204416, -0.19183555, -0.88762896],
                          [-0.74715829,  1.6924546 ,  0.05080775, -0.63699565,  0.19091548]]),np.array([[2.10025514],
                        [0.12015895],
                        [0.61720311]])

t_dA_prev, t_dW, t_db = linear_backward(t_dZ, t_linear_cache)

print("dA_prev: " + str(t_dA_prev))
print("dW: " + str(t_dW))
print("db: " + str(t_db))


# ###### Backward ReLU and Sigmoid activation functions:

# In[21]:


def relu_backward(dA, activation_cache):
    Z = activation_cache
    dZ = np.array(dA, copy=True)  # Just converting dA to a correct object.

    # When Z <= 0, set dZ to 0 as well.
    dZ[Z <= 0] = 0

    return dZ

def sigmoid_backward(dA, activation_cache):
    Z = activation_cache
    s = 1 / (1 + np.exp(-Z))
    dZ = dA * s * (1 - s)

    return dZ


# ##### Implement the backward propagation for the LINEAR->ACTIVATION layer.

# In[22]:


# GRADED FUNCTION: linear_activation_backward

def linear_activation_backward(dA, cache, activation):
    linear_cache, activation_cache = cache

    if activation == "relu":
        dZ = relu_backward(dA, activation_cache)
        dA_prev, dW, db = linear_backward(dZ,linear_cache)

    elif activation == "sigmoid":
        dZ = sigmoid_backward(dA, activation_cache)
        dA_prev, dW, db = linear_backward(dZ,linear_cache)

    return dA_prev, dW, db


# ##### Backward Function: Linear -> ReLU for L-1 layer and Linear -> sigma

# In[36]:


def L_model_backward(AL, Y, caches):

    grads = {}
    L = len(caches) # the number of layers
    m = AL.shape[1]
    Y = Y.reshape(AL.shape) # after this line, Y is the same shape as AL

    dAL = - (np.divide(Y, AL) - np.divide(1 - Y, 1 - AL))
    current_cache = caches[L-1] # Last Layer
    grads["dA" + str(L-1)], grads["dW" + str(L)], grads["db" + str(L)] = linear_activation_backward(dAL, current_cache, "sigmoid")

    for l in reversed(range(L-1)):
        current_cache = caches[l]
        dA_prev_temp, dW_temp, db_temp = linear_activation_backward(grads["dA" + str(l + 1)], current_cache, activation = "relu")
        grads["dA" + str(l)] = dA_prev_temp
        grads["dW" + str(l + 1)] = dW_temp
        grads["db" + str(l + 1)] = db_temp

    return grads


# ##### Updating parameters using gradient descent

# In[37]:


# GRADED FUNCTION: update_parameters

def update_parameters(params, grads, learning_rate):
    parameters = copy.deepcopy(params)
    L = len(parameters) // 2 # number of layers in the neural network
    for l in range(L):
        parameters["W" + str(l+1)] = parameters["W" + str(l+1)] - learning_rate * grads["dW" + str(l+1)]
        parameters["b" + str(l+1)] = parameters["b" + str(l+1)] - learning_rate * grads["db" + str(l+1)]
    return parameters


# #### Preparing Training Data

# #### Loading the npz files that contains training and testing images' Pixel data

# In[45]:


# Load the .npz files
train_data = np.load(r'C:\Users\THANGAMUTHU\Downloads\Data_comparing_dogs_cats\train_data.npz')
test_data = np.load(r'C:\Users\THANGAMUTHU\Downloads\Data_comparing_dogs_cats\test_data.npz')

# List all available keys in the .npz files
print("Keys in train_data:", train_data.files)
print("Keys in test_data:", test_data.files)


# In[46]:


import numpy as np

# Load the .npz files
train_data = np.load(r'C:\Users\THANGAMUTHU\Downloads\Data_comparing_dogs_cats\train_data.npz')
test_data = np.load(r'C:\Users\THANGAMUTHU\Downloads\Data_comparing_dogs_cats\test_data.npz')

# Extract available data
train_x_orig = train_data['train_x_orig']
test_x_orig = test_data['test_x_orig']

# Print shapes to verify
print("train_x_orig shape:", train_x_orig.shape)
print("test_x_orig shape:", test_x_orig.shape)


# In[47]:


# Assuming binary classification (cat vs dog) and only cat images in train
num_train = train_x_orig.shape[0]
num_test = test_x_orig.shape[0]

# Create dummy labels
train_y = np.ones((1, num_train))  # All ones for cat images
test_y = np.zeros((1, num_test))   # Initialize with zeros; modify based on actual test labels if available

# Print shapes to verify
print("train_y shape:", train_y.shape)
print("test_y shape:", test_y.shape)


# In[50]:


print("Keys in test_data:", test_data.files)


# In[49]:


# Paths to the .npz files
train_file_path = r'C:\Users\THANGAMUTHU\Downloads\Data_comparing_dogs_cats\train_data.npz'
test_file_path = r'C:\Users\THANGAMUTHU\Downloads\Data_comparing_dogs_cats\train_data.npz'

# Load the .npz files
train_data = np.load(train_file_path)
test_data = np.load(test_file_path)

# Extract arrays from the .npz files
train_x_orig = train_data['train_x_orig']
test_x_orig = test_data['test_x_orig']

# Create dummy labels
# Number of training and test images
num_train = train_x_orig.shape[0]
num_test = test_x_orig.shape[0]

# Dummy labels for training and test sets
train_y = np.ones((1, num_train))  # All ones for cat images, as the train set contains only cat images
test_y = np.zeros((1, num_test))   # Initialize with zeros; modify as needed if actual test labels are available

# Check if 'classes' is available and extract it if present
if 'classes' in test_data:
    classes = test_data['classes']
else:
    classes = np.array(['cat', 'dog'])  # Default classes if 'classes' is not available

# Verify the shapes of the arrays
print("train_x_orig shape:", train_x_orig.shape)
print("train_y shape:", train_y.shape)
print("test_x_orig shape:", test_x_orig.shape)
print("test_y shape:", test_y.shape)
print("classes shape:", classes.shape if 'classes' in locals() else "Not available")


# In[ ]:


index = 11
plt.imshow(train_x_orig[index])
plt.title("y = " + str(train_y[0,index]) + ". It's a " + classes[train_y[0,index]] + " picture.")
plt.show()


# In[ ]:


# Example of a picture
index = 10

# Display the image
plt.imshow(train_x_orig[index])
plt.title(f"Image index: {index}")
plt.show()

# Make sure train_y is integer
train_y = train_y.astype(int)

# Access label and class
label_index = train_y[0, index]  # Convert to integer index
class_label = classes[label_index].decode("utf-8") if isinstance(classes[0], bytes) else classes[label_index]

# Print the label
print(f"y = {label_index}. It's a {class_label} picture.")


# In[ ]:


# Explore your dataset
m_train = train_x_orig.shape[0]
num_px = train_x_orig.shape[1]
m_test = test_x_orig.shape[0]

print ("Number of training examples: " + str(m_train))
print ("Number of testing examples: " + str(m_test))
print ("Each image is of size: (" + str(num_px) + ", " + str(num_px) + ", 3)")
print ("train_x_orig shape: " + str(train_x_orig.shape))
print ("train_y shape: " + str(train_y.shape))
print ("test_x_orig shape: " + str(test_x_orig.shape))
print ("test_y shape: " + str(test_y.shape))


# In[ ]:


# Reshape the training and test examples
train_x_flatten = train_x_orig.reshape(train_x_orig.shape[0], -1).T   # The "-1" makes reshape flatten the remaining dimensions
test_x_flatten = test_x_orig.reshape(test_x_orig.shape[0], -1).T

# Standardize data to have feature values between 0 and 1.
train_x = train_x_flatten/255.
test_x = test_x_flatten/255.

print ("train_x's shape: " + str(train_x.shape))
print ("test_x's shape: " + str(test_x.shape))


# #### 2 Layer Neural Network

# In[ ]:


### CONSTANTS DEFINING THE MODEL ####
n_x = 12288     # num_px * num_px * 3
n_h = 7
n_y = 1
layers_dims = (n_x, n_h, n_y)
learning_rate = 0.0075


# In[ ]:


# GRADED FUNCTION: two_layer_model

def two_layer_model(X, Y, layers_dims, learning_rate = 0.0075, num_iterations = 3000, print_cost=False):
    np.random.seed(1)
    grads = {}
    costs = []                              # to keep track of the cost
    m = X.shape[1]                           # number of examples
    (n_x, n_h, n_y) = layers_dims

    parameters = initialize_parameters(n_x, n_h, n_y)
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]

    for i in range(0, num_iterations):
        A1, cache1 = linear_activation_forward(X, W1, b1, activation="relu")
        A2, cache2 = linear_activation_forward(A1, W2, b2, activation="sigmoid")

        cost = compute_cost(A2, Y)

        dA2 = - (np.divide(Y, A2) - np.divide(1 - Y, 1 - A2))

        dA1, dW2, db2 = linear_activation_backward(dA2, cache2, activation="sigmoid")
        dA0, dW1, db1 = linear_activation_backward(dA1, cache1, activation="relu")

        grads['dW1'] = dW1
        grads['db1'] = db1
        grads['dW2'] = dW2
        grads['db2'] = db2

        parameters = update_parameters(parameters, grads, learning_rate)

        W1 = parameters["W1"]
        b1 = parameters["b1"]
        W2 = parameters["W2"]
        b2 = parameters["b2"]

        if print_cost and i % 100 == 0 or i == num_iterations - 1:
            print("Cost after iteration {}: {}".format(i, np.squeeze(cost)))
        if i % 100 == 0 or i == num_iterations:
            costs.append(cost)

    return parameters, costs

def plot_costs(costs, learning_rate=0.0075):
    plt.plot(np.squeeze(costs))
    plt.ylabel('cost')
    plt.xlabel('iterations (per hundreds)')
    plt.title("Learning rate =" + str(learning_rate))
    plt.show()


# In[ ]:


parameters, costs = two_layer_model(train_x, train_y, layers_dims = (n_x, n_h, n_y), num_iterations = 2, print_cost=False)

print("Cost after first iteration: " + str(costs[0]))


# #### Training the Model

# In[ ]:


parameters, costs = two_layer_model(train_x, train_y, layers_dims = (n_x, n_h, n_y), num_iterations = 2500, print_cost=True)
plot_costs(costs, learning_rate)


# #### L Layered Neural Network

# In[6]:


### CONSTANTS ###
layers_dims = [12288, 20, 7, 5, 1] #  4-layer model


# #### Implements a L-layer neural network

# In[ ]:


# GRADED FUNCTION: L_layer_model

def L_layer_model(X, Y, layers_dims, learning_rate = 0.0075, num_iterations = 3000, print_cost=False):
    np.random.seed(1)
    costs = []                         # keep track of cost

    parameters = initialize_parameters_deep(layers_dims)

    for i in range(0, num_iterations):
        AL, caches = L_model_forward(X, parameters)
        cost = compute_cost(AL, Y)
        grads = L_model_backward(AL, Y, caches)
        parameters = update_parameters(parameters, grads, learning_rate)
        if print_cost and i % 100 == 0:
            print ("Cost after iteration %i: %f" %(i, cost))
        if print_cost and i % 100 == 0:
            costs.append(cost)
        if print_cost == False and i % 100 == 0:
            costs.append(cost)

    return parameters, costs


# In[ ]:


parameters, costs = L_layer_model(train_x, train_y, layers_dims, num_iterations = 1, print_cost = False)

print("Cost after first iteration: " + str(costs[0]))


# #### Training the model

# In[ ]:


parameters, costs = L_layer_model(train_x, train_y, layers_dims, num_iterations = 2500, print_cost = True)


# #### Predicting the results

# In[ ]:


def predict(X, y, parameters):
    m = X.shape[1]
    n = len(parameters) // 2  # number of layers in the neural network
    predictions = np.zeros((1, m))
    probas, caches = L_model_forward(X, parameters)
    for i in range(probas.shape[1]):
        if probas[0, i] > 0.5:
            predictions[0, i] = 1
        else:
            predictions[0, i] = 0
    print("Accuracy: " + str(np.sum((predictions == y)/m)))

    return predictions


# ##### Checking with any dog or cat image we have

# In[ ]:


my_image = '7.jpg' # Putting here the name of my file that I would like to predict as cat or Dog
my_label_y = [1] # the true class of my image (1 -> cat, 0 -> non-cat)

fname = "images/" + my_image
image = np.array(Image.open(fname).resize((num_px, num_px)))
plt.imshow(image)
image = image / 255.
image = image.reshape((1, num_px * num_px * 3)).T

my_predicted_image = predict(image, my_label_y, parameters)

print("y = " + str(train_y[0,index]) + ". It's a " + str(classes[train_y[0,index]]) + " picture.")

