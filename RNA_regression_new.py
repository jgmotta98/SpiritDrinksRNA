"""
Created on Sun Nov 04 02:15:01 2022
"""
# The algorithm receives training and test data for regression neural network models. It evaluates the model,
# saves it's data into an Excel worksheet and serializes (optional) the model for future use.

# Future: give the exact destination of the saved files, maybe save the X_files to make comparison graphs.

import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, max_error
import pickle

# Preprocessing function to transform and inverse transform input data.
# (MinMax turns the smallest value into 0, and the biggest into 1, adjusting all the values in-between accordingly).
def dataPreprocessing():
    scaler = MinMaxScaler()
    return scaler

# Neural network regression model. Returns the trained model,
# validation variables and all variables evaluation.
def dataRegression(X, y, X_test, y_test):
    X_train, X_validation, y_train, y_validation = train_test_split(X, y, test_size=2/3, random_state=1)

    # Preprocessing data (transforming every training and validation data into a scale from 0 to 1).
    scaler = dataPreprocessing()
    X_train = scaler.fit_transform(X_train)
    X_validation = scaler.transform(X_validation)
    X_test = scaler.transform(X_test)

    rnaModel = MLPRegressor() #Add param later.
    trainedModel = rnaModel.fit(X_train, y_train)

    trainPrediction = trainedModel.predict(X_train)
    validationPrediction = trainedModel.predict(X_validation)
    testPrediction = trainedModel.predict(X_test)

    dataPrediction = [trainPrediction, validationPrediction, testPrediction]

    trainEvaluation= regressionEvaluation(y_train, trainPrediction)
    validationEvaluation = regressionEvaluation(y_validation, validationPrediction)
    testEvaluation = regressionEvaluation(y_test, testPrediction)

    evaluationNames = ['Train', 'Validation', 'Test']

    return trainedModel, y_validation, X_train, y_train, dataPrediction, \
           evaluationNames, trainEvaluation, validationEvaluation, testEvaluation

# All regression evaluation for use. Returns their results.
def regressionEvaluation(y, y_test):
    maxError = max_error(y, y_test)
    mae = mean_absolute_error(y, y_test)
    mse = mean_squared_error(y, y_test)
    r2 = r2_score(y, y_test)
    return maxError, mae, mse, r2

# Receive regression evaluation data. Saves data into an Excel file.
def saveEvaluationToExcel(evaluationList, dataSet):
    oldSamples = pd.read_excel('Parâmetros do RNA.xls')
    # Terrible solution. Fix later.
    dataList = []
    for evaluation, dataName in zip(evaluationList, dataSet):
        dataParam = pd.DataFrame({'Data set': dataName,
                                  'Max error': evaluation[0],
                                  'Mean absolute error': evaluation[1],
                                  'Mean square error': evaluation[2],
                                  'R2': evaluation[3]}, index=[0])
        dataList.append(dataParam)
    dataDf = pd.concat(dataList)
    gatheredData = pd.concat([oldSamples, dataDf])
    gatheredData.to_excel('Parâmetros do RNA.xls', index=False)

# Receive all data information. Saves it into an Excel file.
# (Saves X_train data in a csv file from future preprocessing use).
def saveDataToExcel(y_train, y_validation, y_test, dataPredicted):
    oldSamples = pd.read_excel('Regression_data.xls')
    y_list = [y_train, y_validation, y_test]

    # Saving X_train into a csv file to reuse preprocessing in future analysis.
    trainDf = pd.DataFrame(X_train)
    trainDf.to_csv('raw_train_data.csv', index=False)

    # Terrible solution. Fix later. (Maybe...)
    dataList = []
    for y_data, y_predict in zip(y_list, dataPredicted):
        dataParam = pd.DataFrame({'y_real': y_data,
                                  'y_predict': y_predict})
        dataList.append(dataParam)
    dataDf = pd.concat(dataList)
    gatheredData = pd.concat([oldSamples, dataDf])
    gatheredData.to_excel('Regression_data.xls', index=False)

# Save trained RNA structure as a pickle file (series of bytes that can be converted back into python code)
# for reproducibility in later uses.
def serializingRnaStructure(rnaStructureName, rnaStructureVariable):
    with open(rnaStructureName+'.pickle', 'wb') as f:
        pickle.dump(rnaStructureVariable, f)

# Importing samples from Excel files and treating data.
trainingSamples = pd.read_excel("amostratreinamento.xlsx")
testSamples = pd.read_excel("amostrateste.xlsx")

X = trainingSamples.iloc[:,1:-1]
y = trainingSamples.iloc[:-1]

X_test = testSamples.iloc[:,1:-1]
y_test = testSamples.iloc[:,-1]

# Data regression (Model, X_validation, y_validation,
# respective data set names, evaluation results).
trainedModel, y_validation, X_train, y_train, dataPrediction, dataSet, y_trainEvaluation, \
y_validationEvaluation, y_testEvaluation = dataRegression(X, y, X_test, y_test)
evaluationList = [y_trainEvaluation, y_validationEvaluation, y_testEvaluation]

# Saving prediction data into an Excel worksheet.
saveDataToExcel(X_train, y_train, y_validation, y_test, dataPrediction)

# Saving evaluation data into an Excel worksheet.
saveEvaluationToExcel(evaluationList, dataSet)

# Save as pickle file.
serializingRnaStructure('RegressionSample', trainedModel)