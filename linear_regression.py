import numpy as np
import matplotlib.pyplot as plt


def initialize_parameters(lenw):
    w = np.random.randn(1, lenw)
    b = 0
    return w,b

def forward_prop(X,w,b):
    z = np.dot(w, X) + b
    return z

def cost_function(z,y):
    m = y.shape[1]
    J = (1/(2*m)) * np.sum(np.sqaure(z-y))
    return J

def back_prop(X,y,z):
    m = y.shape[1]
    dz = (1/m) * (z-y)
    dw = np.dot(dz, X.T)
    db = np.sum(dz)
    return dw,db


def gradient_descent_update(w, b, dw, db, learning_rate):
    w = w - learning_rate*dw
    b = b - learning_rate*db
    return w,b

def linear_regression_model(X_train, y_train, X_val, y_val, learning_rate, epochs):
    lenw = X_train.shape[0]
    w,b = initialize_parameters(lenw)

    costs_train = []
    m_train = y_train.shape[1]
    m_val = y_val.shape[1]

    for i in range(1, epochs+1):
        z_train = forward_prop(X_train, w, b)
        cost_train = cost_function(z_train, y_train)
        dw,db = back_prop(X_train,y_train,z_train)
        w,b = gradient_descent_update(w, b, dw, db, learning_rate)

        if i%10 == 0:
            costs_train.append(cost_train)

        MAE_train = (1/m_train)*np.sum(np.abs(z_train-y_train))

        z_val = forward_prop(X_val,w,b)
        cost_val = cost_function(z_val, y_val)
        MAE_val = (1 / m_val) * np.sum(np.abs(z_val - y_val))

        print("Epoch:" + str(i) + "\n\tCost: "+ str(cost_train) + "\n\tValidation cost: " + str(cost_val) + "\n\tMAE cost: "+ str(MAE_train) + "\n\tMAE Validation cost: " + str(MAE_val))

    plt.plot(cost_train)
    plt.xlabel("Iterations by 10's")
    plt.ylabel("Training Cost")
    plt.title("Learning rate " + str(learning_rate))
    plt.show()