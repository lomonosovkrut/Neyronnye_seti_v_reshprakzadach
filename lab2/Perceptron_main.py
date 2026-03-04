# -*- coding: utf-8 -*-
"""
Основной скрипт для тестирования многослойного перцептрона
"""
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import pandas as pd
import numpy as np
from neural import Perceptron

# Загрузка и подготовка данных
df = pd.read_csv('data.csv')
df = df.iloc[np.random.permutation(len(df))]

# Первые 100 примеров для обучения
y_train = df.iloc[0:100, 4].values
y_train = np.where(y_train == "Iris-setosa", 1, -1)
X_train = df.iloc[0:100, [0, 2]].values

# Все данные для тестирования
y_test = df.iloc[:, 4].values
y_test = np.where(y_test == "Iris-setosa", 1, -1)
X_test = df.iloc[:, [0, 2]].values

# Параметры сети
inputSize = X_train.shape[1]
hiddenSizes = [10, 10]  
outputSize = 1

# Создание и обучение сети
NN = Perceptron(inputSize, hiddenSizes, outputSize)
NN.train(X_train, y_train, n_iter=500, eta=0.01)

# Тестирование
out, hidden = NN.predict(X_test)
errors = np.sum(out.flatten() != y_test)
print(f"Количество ошибок на тестовой выборке: {errors}")
print(f"Точность: {(len(y_test) - errors) / len(y_test) * 100:.2f}%")