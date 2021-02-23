import numpy as np
import math as m
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
a = np.loadtxt('dane15.txt')

x = a[:,[0]]    #INPUT
y = a[:,[1]]    #OUTPUT

X_train, X_test, y_train, y_test = train_test_split(x,y, test_size=0.3, random_state=1)

x=x.T
y=y.T

#Sieć
S1= np.count_nonzero(x)
W1 = np.random.rand(S1,1) - 0.5
B1 = np.random.rand(S1,1) - 0.5
W2 = np.random.rand(1,S1) - 0.5
B2 = np.random.rand(1,1) - 0.5
lr = 0.001

#odpowiedź sieci metoda wsadowa
for i in range(100):
    X = W1 @ x + B1 @ np.ones(x.shape)
    A1 = np.maximum(X,0)
    A2 = W2 @ A1 + B2

#propagacja wsteczna
    E2 = y - A2
    E1 = W2.T * E2

    dW2 = lr * E2 @ A1.T
    dB2 = lr * E2 @ np.ones(E2.shape).T
    dW1 = lr * (np.exp(X)/(np.exp(X) + 1)) * E1 @ x.T 
    dB1 = lr * (np.exp(X)/(np.exp(X) + 1)) * E1 @ np.ones(x.shape).T

    W2 = W2 + dW2
    B2 = B2 + dB2
    W1 = W1 + dW1
    B1 = B1 + dB1

    if (i % 1==0):
        
        plt.plot(x,y,'g^')
        plt.plot(x[0], A2[0])
        plt.pause(0.00001)
        plt.clf()
    
 
