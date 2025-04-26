import math
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns

np.set_printoptions(precision=9, suppress=True)

df = pd.read_csv("kc_house_data.csv")
df = df.dropna()

X = df[["bedrooms", "sqft_lot", "sqft_living", "floors", "view", "yr_built", "lat", "long"]].values
X = (X - X.mean(axis = 0))/X.std(axis=0)

y = df[["price"]].values

XTrain, XTest, yTrain, yTest = train_test_split(X, y, test_size=0.3, shuffle=True)

class NeuralNetwork:
    def __init__(self, nFeatures):
        self.weights = np.random.randn(nFeatures + 1)

    def Predict(self, xs):
        return self.weights[1:] @ xs.T + self.weights[0]

    def Optimize(self, xs, y, prediction, batchSize, learningRate):
        error = prediction - y.T
        gradWeights = 2*(error @ xs)/batchSize
        gradBias    = 2*np.mean(error)

        self.weights[1:] -= learningRate*gradWeights.reshape(xs.shape[1],)
        
        self.weights[0]  -= learningRate*gradBias
    
    def Train(self, xs, y, batchSize, learningRate):
        prediction = self.Predict(xs)
        self.Optimize(xs, y, prediction, batchSize, learningRate)
        return np.mean((prediction - y) ** 2)
    
    def Fit(self, epochs, batchSize, learningRate):  
        trainHistory = []
        testHistory  = []
        for epoch in range(epochs):
            indices = np.random.permutation(len(XTrain))
            XTrainShuffled = XTrain[indices]
            yTrainShuffled = yTrain[indices]
            
            trainLoss = 0
            for i in range(0, len(features), batchSize):
                xsBatch = XTrainShuffled[i:i+batchSize]
                yBatch  = yTrainShuffled[i:i+batchSize]
                if (len(xsBatch) == 0):
                    continue
                trainLoss += self.Train(xsBatch, yBatch, len(xsBatch), learningRate)
            trainLoss /= (len(XTrain)//batchSize)

            testPrediction = self.Predict(XTest)
            testLoss = np.mean((testPrediction - yTest)**2)

            trainHistory.append(math.sqrt(trainLoss))
            testHistory.append(math.sqrt(testLoss))
            
            if ((epoch+1)%(epochs*0.05) == 0):
                print(f"[{epoch+1}/{epochs}] Train Loss: {math.sqrt(trainLoss)}, Test Loss: {math.sqrt(testLoss)}")
            if ((epoch+1)%(epochs*0.2) == 0):
                plt.plot(trainHistory)
                plt.plot(testHistory)
                plt.show()
            
model = NeuralNetwork(XTrain.shape[1])
model.Fit(epochs=300, batchSize=8, learningRate=0.0000025)
