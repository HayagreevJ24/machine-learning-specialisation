# Implementation of univariate linear regression from scratch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import random

# Preprocessing of data

df = pd.read_csv('chip_dataset.csv')
df = df.dropna()
df.info()
xlabel = 'Transistors (million)'
ylabel = 'Die Size (mm^2)'
arrayOfTarget = df[xlabel].to_numpy()
arrayOfFeature = df[ylabel].to_numpy()
m = len(arrayOfFeature)

# Normalising the dataset
featureMean, targetMean = np.mean(arrayOfFeature), np.mean(arrayOfTarget)
featuresd, targetsd = np.std(arrayOfFeature), np.std(arrayOfTarget)

for i in range(m):
    arrayOfFeature[i] = (arrayOfFeature[i] - featureMean)/featuresd
    arrayOfTarget[i] = (arrayOfTarget[i] - targetMean)/targetsd


def calculate_cost(w, b):
    costFunction = 0
    for (x, y) in zip(arrayOfFeature, arrayOfTarget):
        costFunction += ((w * x) + b - y) ** 2
    costFunction = costFunction / (2 * m)
    return costFunction


def partialDerivativeW(w, b):
    partialDerivatve = 0
    for (x, y) in zip(arrayOfFeature, arrayOfTarget):
        partialDerivatve += ((w * x + b) - y) * x

    partialDerivatve = partialDerivatve / m
    return partialDerivatve


def partialDerivativeB(w, b):
    partialDerivative = 0
    for (x, y) in zip(arrayOfFeature, arrayOfTarget):
        partialDerivative += ((w * x + b) - y)
    partialDerivative = partialDerivative / m
    return partialDerivative


def adjustParameters(w, b, learningRate=0.01):
    w_new = w - learningRate * partialDerivativeW(w, b)
    b_new = b - learningRate * partialDerivativeB(w, b)
    return w_new, b_new


def linearRegressionModel():
    w = random.random()
    b = 0
    learningRate = 0.01
    previousCost = calculate_cost(w, b)

    while True:
        w, b = adjustParameters(w, b, learningRate)
        currentCost = calculate_cost(w, b)

        # Check for convergence - very small difference in cost
        if abs(previousCost - currentCost) < 1e-6:
            break
        previousCost = currentCost

    r = np.sum((arrayOfFeature - np.mean(arrayOfFeature)) / (np.std(arrayOfFeature)) * (arrayOfTarget - np.mean(arrayOfTarget) / np.std(arrayOfTarget))) / (m - 1)
    rsquared = r ** 2

    modelMetrics = {'slope': w, 'intercept': b, 'MSE': calculate_cost(w, b), 'correlation': r, 'coefficientOfDetermination': rsquared}
    return modelMetrics

def trainModel():
    modelMetrics = linearRegressionModel()
    return modelMetrics

def predict(predictor, model):
    normalisedpredictor = (predictor - featureMean)/featuresd
    normalisedprediction = model['slope'] * normalisedpredictor + model['intercept']
    prediction = (normalisedprediction * targetsd) + targetMean
    return prediction

def main():
    chipdataModelMetrics = trainModel()
    print(f"... Regression with gradient descent complete!")
    print(f"Slope: {chipdataModelMetrics['slope']}")
    print(f"Intercept: {chipdataModelMetrics['intercept']}")
    print(f"Mean squared error: {chipdataModelMetrics['MSE']}")
    print(f"Strength of relationship: {chipdataModelMetrics['correlation']}")
    print(f"Coefficient of determination: {chipdataModelMetrics['coefficientOfDetermination']}")

    # Plotting the data on a scatter plot.
    plt.scatter(arrayOfFeature, arrayOfTarget, c='blue')
    arrayOfPredictions = np.zeros(m)
    for k in range(m):
        arrayOfPredictions[k] = arrayOfFeature[k] * chipdataModelMetrics['slope'] + chipdataModelMetrics['intercept']

    plt.xlabel("Normalised " + xlabel)
    plt.ylabel("Normalised " + ylabel)
    plt.plot(arrayOfFeature, arrayOfPredictions, color='red', label='Regression Line')
    plt.show()

    print(f"The predicted Die size (mm\u00b2) for 100 million transistors is {predict(100, chipdataModelMetrics)}")

main()