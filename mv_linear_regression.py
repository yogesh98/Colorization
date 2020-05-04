import numpy as np
import matplotlib.pyplot as plt
from numpy import square as sq

class LinearRegression:

    def __init__(self, X_train, y_train, X_val, y_val, learning_rate, epochs):
        lenw = X_train.shape[0]
        self.w = np.random.randn(1,lenw)
        self.b = 0
        self.train(X_train, y_train, X_val, y_val, learning_rate, epochs)

    def forward_prop(self,X):
        z = np.dot(self.w, X) + self.b
        return z

    def cost_function(self,z,y):
        m = y.shape[1]
        J = (1 / (2 * m)) * np.sum(sq(z - y))
        return J

    def back_prop(self, X, y, z):
        m = y.shape[1]
        dz = (1 / m) * (z - y)
        dw = np.dot(dz, X.T)
        db = np.sum(dz)
        return dw, db

    def gradient_descent_upate(self,dw,db, learning_rate):
        self.w = self.w - learning_rate*dw
        self.b = self.b - learning_rate*db

    def train(self, X_train, y_train, X_val, y_val, learning_rate, epochs):

        costs_train = []
        m_train = y_train.shape[1]
        m_val = y_val.shape[1]

        for i in range(1, epochs+1):
            z_train = self.forward_prop(X_train)
            cost_train = self.cost_function(z_train, y_train)
            dw,db = self.back_prop(X_train,y_train,z_train)
            self.gradient_descent_upate(dw,db,learning_rate)

            if i % 10 == 0:
                costs_train.append(cost_train)

            MAE_train = (1/m_train)*np.sum(np.abs(z_train-y_train))

            z_val = self.forward_prop(X_val)
            cost_val = self.cost_function(z_val, y_val)
            MAE_val = (1 / m_val) * np.sum(np.abs(z_val - y_val))

            print("Epoch:" + str(i) + "\n\tCost: "+ str(cost_train) + "\n\tValidation cost: " + str(cost_val) + "\n\tMAE cost: "+ str(MAE_train) + "\n\tMAE Validation cost: " + str(MAE_val))

        plt.plot(costs_train)
        plt.xlabel("Iterations by 10's")
        plt.ylabel("Training Cost")
        plt.title("Learning rate " + str(learning_rate))
        plt.show()
