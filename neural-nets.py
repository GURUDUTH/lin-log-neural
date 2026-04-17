#Forward propagation -> single layer
#with input of two features , a hidden layer with 3 activation function units [a1]
#and an ouput layer with a single neuron [a2]


import numpy as np

def sigmoid(z):
    g = 1/(1+np.exp(-z))
    return g

x = np.array([200,17])

#Hidden layer with 3 activation units
w1_1 = np.array([1,2])
b1_1 = np.array([-1])
z1_1 = np.dot(w1_1,x) + b1_1
a1_1 = sigmoid(z1_1)
print(a1_1)


w1_2 = np.array([-3,4])
b1_2 = np.array([1])
z1_2 = np.dot(w1_2,x) + b1_2
a1_2 = sigmoid(z1_2)
print(a1_2)


w1_3 = np.array([5,-6])
b1_3 = np.array([2])
z1_3 = np.dot(w1_3,x) + b1_3
a1_3 = sigmoid(z1_3)
print(a1_3)

a1 = np.array([a1_1,a1_2,a1_3])


#output layer single activation unit
w2_1 = np.array([-7,8,9])
b2_1 = np.array([3])
z2_1 = np.dot(w2_1,a1) + b2_1
a2   = sigmoid(z2_1)
print(a2)


"""instead of writing the ctivation function for each units in each layer we can use a more
General implementation of forward propagation"""

x = np.array([-2,4])

W = np.array([[1,-3,5],
              [2,4,-6]])

b = np.array([-1,1,2])

#defining the sigmoid
def sigmoid(z):
    g = 1/(1+np.exp(-z))
    return g

#defining the activation loop for single hidden layer with 3 activation units
def hidden_layer(x):
    units = W.shape[1]
    l1_out = np.zeros(units) #first layer named as a1
    for i in range(units):
        z = np.dot(W[:,i],x)+b[i]
        g = sigmoid(z)
        l1_out[i] = g #layer one output
    return l1_out

a_1 = hidden_layer(x)
print(a_1)

"""we can use the defined function for one hidden layer and caryy computation for
many varying layer of the same Neural Network"""

def sequential(x):
    a1 = hidden_layer(x,W1,b1)
    a2 = hidden_layer(a1,W2,b2) 
    a3 = hidden_layer(a2,W3,b3) 
    a4 = hidden_layer(a3,W4,b4)
    f_x = a4
    return f_x

result = sequential(x)
print(result)