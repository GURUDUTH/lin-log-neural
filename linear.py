#Forward propagation -> single layer
#with input of two features , a hidden layer with 3 neuron units [a1]
#and an ouput layer with a single neuron [a2]


import numpy as np

def sigmoid(z):
    g = 1/(1+np.exp(-z))
    return g

x = np.array([200,17])

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


w2_1 = np.array([-7,8,9])
b2_1 = np.array([3])
z2_1 = np.dot(w2_1,a1) + b2_1
a2   = sigmoid(z2_1)
print(a2)





W = np.array([[1,-3,5],
               [2,4,-6]])
print(W.shape[1])       
print(W[:,0])

