# -*- coding: utf-8 -*-
"""
Многослойный перцептрон с произвольным количеством скрытых слоёв
"""
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import numpy as np

class Perceptron:
    def __init__(self, inputSize, hiddenSizes, outputSize):
        """
        inputSize: количество входных нейронов
        hiddenSizes: список количества нейронов в каждом скрытом слое
        outputSize: количество выходных нейронов
        """
        self.inputSize = inputSize
        self.hiddenSizes = hiddenSizes
        self.outputSize = outputSize
        self.n_hidden_layers = len(hiddenSizes)
        
        # Инициализация весов для всех слоёв
        self.weights = []
        
        # Первый скрытый слой
        W = np.zeros((1 + inputSize, hiddenSizes[0]))
        W[0, :] = np.random.randint(0, 3, size=hiddenSizes[0])
        W[1:, :] = np.random.randint(-1, 2, size=(inputSize, hiddenSizes[0]))
        self.weights.append(W)
        
        # Промежуточные скрытые слои
        for i in range(1, self.n_hidden_layers):
            W = np.zeros((1 + hiddenSizes[i-1], hiddenSizes[i]))
            W[0, :] = np.random.randint(0, 3, size=hiddenSizes[i])
            W[1:, :] = np.random.randint(-1, 2, size=(hiddenSizes[i-1], hiddenSizes[i]))
            self.weights.append(W)
        
        # Выходной слой
        W = np.random.randint(0, 2, size=(1 + hiddenSizes[-1], outputSize)).astype(np.float64)
        self.weights.append(W)
    
    def predict(self, Xp):
        """Прямой проход через сеть"""
        # Обработка одного или нескольких примеров
        if Xp.ndim == 1:
            Xp = Xp.reshape(1, -1)
        
        # Проход через все скрытые слои
        hidden = Xp
        for i in range(self.n_hidden_layers):
            W = self.weights[i]
            hidden = np.where((np.dot(hidden, W[1:, :]) + W[0, :]) >= 0.0, 1, -1).astype(np.float64)
        
        # Выходной слой
        Wout = self.weights[-1]
        out = np.where((np.dot(hidden, Wout[1:, :]) + Wout[0, :]) >= 0.0, 1, -1).astype(np.float64)
        
        return out, hidden
    
    def train(self, X, y, n_iter=5, eta=0.01):
        """Обучение сети (обучаются только веса выходного слоя)"""
        for i in range(n_iter):
            for xi, target in zip(X, y):
                # Прямой проход для одного примера
                pr, hidden = self.predict(xi)
                
                # Исправление форм для корректного broadcasting
                # pr имеет форму (1, 1), target - скаляр
                error = target - pr.flatten()[0]  # скаляр
                
                # hidden имеет форму (1, h2), нужно (h2,) для умножения
                hidden_flat = hidden.flatten()  # (h2,)
                
                # Обновление весов выходного слоя
                # self.weights[-1][1:] имеет форму (h2, outputSize)
                # error * hidden_flat имеет форму (h2,)
                self.weights[-1][1:, :] += (eta * error * hidden_flat).reshape(-1, 1)
                
                # Обновление порога (bias)
                self.weights[-1][0, :] += eta * error
        
        return self