import math
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns

np.set_printoptions(precision=9, suppress=True)
np.random.seed(1337)

df = pd.read_csv("kc_house_data.csv")
df = df.dropna()

X = df[["bedrooms", "bathrooms", "sqft_living", "sqft_lot", "floors", "waterfront", "view", "condition", "grade", "sqft_above", "sqft_basement", "yr_built", "yr_renovated", "zipcode", "lat", "long", "sqft_living15", "sqft_lot15"]].values
y = df[["price"]].values
XTrain, XTest, yTrain, yTest = train_test_split(X, y, test_size=0.2, random_state=1337)

mean, std = XTrain.mean(axis=0), XTrain.std(axis=0)
XTrain = (XTrain - mean) / std
XTest  = (XTest  - mean) / std

print(df.columns)

class NeuralNetwork:
    def __init__(self, nFeatures):
        self.weights = np.random.randn(nFeatures + 1)

    def Predict(self, xs):
        return (self.weights[1:] @ xs.T + self.weights[0]).reshape(-1, 1)

    def Optimize(self, xs, y, prediction, batchSize, learningRate):
        error = prediction - y
        gradWeights = (2*(error.T @ xs)/batchSize).flatten()
        gradBias    = 2*np.mean(error)

        self.weights[1:] -= learningRate*gradWeights
        self.weights[0]  -= learningRate*gradBias
    
    def Train(self, xs, y, batchSize, learningRate):
        prediction = self.Predict(xs)
        self.Optimize(xs, y, prediction, batchSize, learningRate)
        return np.sum((prediction - y)**2)
    
    def Fit(self, epochs, batchSize, learningRate):  
        trainHistory = []
        testHistory  = []
        for epoch in range(epochs):
            indices = np.random.permutation(len(XTrain))
            XTrainShuffled = XTrain[indices]
            yTrainShuffled = yTrain[indices]
            
            trainLoss = 0
            for i in range(0, len(XTrainShuffled), batchSize):
                xsBatch = XTrainShuffled[i:i+batchSize]
                yBatch  = yTrainShuffled[i:i+batchSize]
                if (len(xsBatch) == 0):
                    continue
                trainLoss += self.Train(xsBatch, yBatch, len(xsBatch), learningRate)

            trainLoss /= len(XTrain)

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
model.Fit(epochs=3000, batchSize=16, learningRate=0.000005)
