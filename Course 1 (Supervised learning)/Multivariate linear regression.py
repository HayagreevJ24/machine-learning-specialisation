# An implementation of multivariate linear regression from scratch
# Plan for implementation in code
"""
Feautres: age, creatinine_phosphokinase, serum_creatinine, serum_sodium, platelets -> 5 prediction features (x1, x2, x3, x4, x5) [n = 5]
Target: ejection_fraction
Number of training examples: 299
Blueprint of modules:
1. Initialise parameters Weights: [w1, w2, w3, w4, w5], b
2. Calculating cost function
3. Calculating derivatvies with respect to each of the parameters
4. Update the parameters
5. Prediction function
6. Data processing - Splitting into train and test, feature scaling.
7. Accuracy evalutation - Using the testing data on the model for predictions and making

"""

# Imports
import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# Data preprocessing
df = pd.read_csv('heart_failure_clinical_records_dataset.csv')
df.dropna(inplace=True)
df.info()
n = 5
numberOfElements = 299
m = int(0.70 * numberOfElements)
o = numberOfElements - m  # Number of testing Examples

# Putting all the data into numpy arrays.
x1master = df['age'].to_numpy()
x2master = df['creatinine_phosphokinase'].to_numpy()
x3master = df['serum_creatinine'].to_numpy()
x4master = df['serum_sodium'].to_numpy()
x5master = df['platelets'].to_numpy()
ymaster = df['ejection_fraction'].to_numpy()


# Populating all the data into required dictionaries of this format
# x(n) = {trainArr:[], testArr[], trainArrMean: , trainArrStd: , trainArrFeatureScaled: []}

def populateDictionary(masterArray):
    trainArr = masterArray[0: m]
    testArr = masterArray[m: numberOfElements]
    trainArrMean = np.mean(trainArr)
    trainArrSTD = np.std(trainArr)
    trainArrFeatureScaled = (trainArr - trainArrMean) / trainArrSTD
    return {
        'trainArr': trainArr,
        'testArr': testArr,
        'trainArrMean': trainArrMean,
        'trainArrSTD': trainArrSTD,
        'trainArrFeatureScaled': trainArrFeatureScaled
    }


x1dict = populateDictionary(x1master)
x2dict = populateDictionary(x2master)
x3dict = populateDictionary(x3master)
x4dict = populateDictionary(x4master)
x5dict = populateDictionary(x5master)
ydict = populateDictionary(ymaster)


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Initialising parameters using vectorisation for the weights
def initialiseParameters():
    w = np.array([2 * random.random() - 1 for _ in range(n)])  # Initialize weights between -1 and 1
    b = random.random()  # Initialise a random bias
    return w, b


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def computeCost(w: np.ndarray, b: float) -> float:
    xvectors = np.column_stack((x1dict['trainArrFeatureScaled'], x2dict['trainArrFeatureScaled'],
                                x3dict['trainArrFeatureScaled'], x4dict['trainArrFeatureScaled'],
                                x5dict['trainArrFeatureScaled']))
    predictions = np.dot(xvectors, w) + b
    errors = predictions - ydict['trainArrFeatureScaled']
    lossFunction = 0.5 * np.sum(errors ** 2)
    costFunction = lossFunction / m
    return costFunction
    # xvectors is of dimension (m, n), predictions is of (m,) [b is broadcast to this] and vectorised operations are used for faster implementation.
    # The same methodology is applied to calculateDeviatives


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def calculateDerivativesW(w: np.ndarray, b: float, featureNumber: int) -> float:
    x_vectors = np.column_stack((x1dict['trainArrFeatureScaled'], x2dict['trainArrFeatureScaled'],
                                 x3dict['trainArrFeatureScaled'], x4dict['trainArrFeatureScaled'],
                                 x5dict['trainArrFeatureScaled']))
    predictions = np.dot(x_vectors, w) + b
    errors = predictions - ydict['trainArrFeatureScaled']
    derivatives = np.dot(errors, x_vectors[:, featureNumber])
    return derivatives / m


def calculateDerivativesB(w: np.ndarray, b: float) -> float:
    x_vectors = np.column_stack((x1dict['trainArrFeatureScaled'], x2dict['trainArrFeatureScaled'],
                                 x3dict['trainArrFeatureScaled'], x4dict['trainArrFeatureScaled'],
                                 x5dict['trainArrFeatureScaled']))
    predictions = np.dot(x_vectors, w) + b
    errors = predictions - ydict['trainArrFeatureScaled']
    derivatives = np.sum(errors)
    return derivatives / m


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def updateParameters(w, b, learningRate=0.1):
    w_new = np.zeros(n)
    for i in range(n):
        w_new[i] = w[i] - learningRate * calculateDerivativesW(w, b, i)

    b_new = b - learningRate * calculateDerivativesB(w, b)
    return w_new, b_new


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def TrainModel():
    w, b = initialiseParameters()
    iterationLimit = 1000
    iterationCount = 0
    previousCost = computeCost(w, b)
    # To plot the learning curve for the data and check if learning rate is appropriate
    currentCostArr = []
    while iterationCount < iterationLimit:
        w, b = updateParameters(w, b)
        currentCost = computeCost(w, b)
        iterationCount += 1
        print(f'Iteration Number: {iterationCount}, Current Cost: {currentCost}, Previous Cost: {previousCost}')
        currentCostArr.append(currentCost)
        if abs(previousCost - currentCost) <= 1e-6:
            break

        previousCost = currentCost

    iterationArray = np.arange(iterationCount)

    modelMetrics = {'w': w, 'b': b, 'MSE': currentCost}
    return modelMetrics, currentCostArr, iterationArray


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def predict(modelMetrics, x1pred, x2pred, x3pred, x4pred, x5pred):
    x1pred = (x1pred - x1dict['trainArrMean']) / x1dict['trainArrSTD']
    x2pred = (x2pred - x2dict['trainArrMean']) / x2dict['trainArrSTD']
    x3pred = (x3pred - x3dict['trainArrMean']) / x3dict['trainArrSTD']
    x4pred = (x4pred - x4dict['trainArrMean']) / x4dict['trainArrSTD']
    x5pred = (x5pred - x5dict['trainArrMean']) / x5dict['trainArrSTD']

    xpredvector = np.array([x1pred, x2pred, x3pred, x4pred, x5pred])
    ypred = (np.dot(xpredvector, modelMetrics['w']) + modelMetrics['b']) * ydict['trainArrSTD'] + ydict['trainArrMean']
    return ypred


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def evaluateModel(modelMetrics):
    # Caculates MSE, RMSE, MAE, r and r^2 with the test data from sklearn module.
    ytest = []
    yactual = ydict['testArr']
    for j in range(o):
        x1test = x1dict['testArr'][j]
        x2test = x2dict['testArr'][j]
        x3test = x3dict['testArr'][j]
        x4test = x4dict['testArr'][j]
        x5test = x5dict['testArr'][j]
        ytestvalue = predict(modelMetrics, x1test, x2test, x3test, x4test, x5test, )
        ytest.append(ytestvalue)

    evaluatedModelMetrics = {
        'MSE': mean_squared_error(yactual, ytest),
        'MAE': mean_absolute_error(yactual, ytest),
        'RMSE': mean_squared_error(yactual, ytest) ** (1 / 2),
        'coeffOfDeterminantion': r2_score(yactual, ytest),
        'pearsonCoefficient': r2_score(yactual, ytest)
    }

    return evaluatedModelMetrics


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def main():
    modelMetrics, costArr, iterationsArr = TrainModel()
    evaluatedModelMetrics = evaluateModel(modelMetrics)

    print(f"Model Metrics: {modelMetrics}")
    print(f"Results from test data: {evaluatedModelMetrics}")

    # Plotting the learning curve
    plt.plot(iterationsArr, costArr)
    plt.xlabel('Number Of iterations')
    plt.ylabel('Cost function')
    plt.title('Learning curve')
    plt.show()


main()