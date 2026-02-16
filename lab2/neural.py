import numpy as np

class Perceptron:
    def __init__(self, inputSize, hiddenSizes, outputSize):
        
      

        self.Win = np.zeros((1+inputSize,h1))
        self.Win[0,:] = (np.random.randint(0, 3, size = (h1)))
        self.Win[1:,:] = (np.random.randint(-1, 2, size = (inputSize,h1)))
        
        self.Win2 = np.zeros((1+h1,h2))
        self.Win2[0,:] = (np.random.randint(0, 3, size = (h2)))
        self.Win2[1:,:] = (np.random.randint(-1, 2, size = (h1,h2)))
        self.Wout = np.random.randint(0, 2, size = (1+h2,outputSize)).astype(np.float64)
        
        
    def predict(self, Xp):
        hidden_predict = np.where((np.dot(Xp, self.Win[1:,:]) + self.Win[0,:]) >= 0.0, 1, -1).astype(np.float64)
        hidden_predict2 = np.where((np.dot(hidden_predict, self.Win2[1:,:]) + self.Win2[0,:]) >= 0.0, 1, -1).astype(np.float64)
        out = np.where((np.dot(hidden_predict2, self.Wout[1:,:]) + self.Wout[0,:]) >= 0.0, 1, -1).astype(np.float64)
        return out, hidden_predict2

    def train(self, X, y, n_iter=5, eta = 0.01):
        for i in range(n_iter):
            print(self.Wout.reshape(1, -1))
            for xi, target, j in zip(X, y, range(X.shape[0])):
                pr, hidden = self.predict(xi)
                self.Wout[1:] += ((eta * (target - pr)) * hidden).reshape(-1, 1)
                self.Wout[0] += eta * (target - pr)
        return self