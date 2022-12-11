"""
Created on Sun Nov 04 02:15:01 2022
"""
# The algorithm receives training and test data to create a MLP classification neural network model. It
# evaluates the model, saves it's data into an Excel worksheet and serializes (optional) the model for future use.

# Future: give the exact destination of the saved files, maybe save the X_files to make comparison graphs.

import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import recall_score, precision_score, f1_score, log_loss, roc_auc_score, confusion_matrix
import pickle
import seaborn as sns
import matplotlib.pyplot as plt

# Preprocessing function to transform and inverse transform input data.
# (MinMax turns the smallest value into 0, and the biggest into 1, adjusting all the values in-between accordingly).
def dataPreprocessing():
    scaler = MinMaxScaler()
    return scaler

# Neural network classification model. Returns the trained model,
# validation variables and all variables evaluation.
def dataClassification(X, y, X_test, y_test):
    X_train, X_validation, y_train, y_validation = train_test_split(X, y, test_size=2 / 3, random_state=1)

    # Preprocessing data (transforming every training and validation data into a scale from 0 to 1).
    scaler = dataPreprocessing()
    X_train = scaler.fit_transform(X_train)
    X_validation = scaler.transform(X_validation)
    X_test = scaler.transform(X_test)

    rnaModel = MLPClassifier() #Add param later.
    trainedModel = rnaModel.fit(X_train, y_train)

    trainPrediction = trainedModel.predict(X_train)
    validationPrediction = trainedModel.predict(X_validation)
    testPrediction = trainedModel.predict(X_test)

    dataPrediction = [trainPrediction, validationPrediction, testPrediction]

    trainEvaluation = classificationEvaluation(y_train, trainPrediction)
    validationEvaluation = classificationEvaluation(y_validation, validationPrediction)
    testEvaluation = classificationEvaluation(y_test, testPrediction)

    EvaluationNames = ['Train', 'Validation', 'Test']

    return trainedModel, y_validation, X_train, y_train, dataPrediction, \
           EvaluationNames, trainEvaluation, validationEvaluation, testEvaluation

# All classification evaluation for use. Returns their results.
def classificationEvaluation(y, y_test):
    recall = recall_score(y, y_test)
    precision = precision_score(y, y_test)
    roc = roc_auc_score(y, y_test)
    f1 = f1_score(y, y_test)
    loss = log_loss(y, y_test)
    confMatrix = confusion_matrix(y, y_test)
    return recall, precision, roc, f1, loss, confMatrix

# Receive classification evaluation data. Saves data into an Excel file.
def saveEvaluationToExcel(evaluationList, dataSet):
    oldSamples = pd.read_excel('Parameters_of_RNA_Classification.xls')
    # Terrible solution. Fix later.(Maybe...)
    dataList = []
    for evaluation, dataName in zip(evaluationList, dataSet):
        dataParam = pd.DataFrame({'Data set': dataName,
                                  'Recall': evaluation[0],
                                  'Precision': evaluation[1],
                                  'ROC': evaluation[2],
                                  'F1': evaluation[3],
                                  'Loss': evaluation[4]}, index=[0])
        dataList.append(dataParam)
    dataDf = pd.concat(dataList)
    gatheredData = pd.concat([oldSamples, dataDf])
    gatheredData.to_excel('Parameters_of_RNA_Classification.xls', index=False)

# Receive all data information. Saves it into an Excel file.
# (Saves X_train data in a csv file from future preprocessing use).
def saveDataToExcel(X_train, y_train, y_validation, y_test, dataPredicted):
    oldSamples = pd.read_excel('Classification_data.xls')
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
    gatheredData.to_excel('Classification_data.xls', index=False)

# Save trained RNA structure as a pickle file (series of bytes that can be converted back into python code)
# for reproducibility in later uses.
def serializingRnaStructure(rnaStructureName, rnaStructureVariable):
    with open(rnaStructureName + '.pickle', 'wb') as f:
        pickle.dump(rnaStructureVariable, f)

# Returns 2d confusion matrix plot.
def confusionMatrixPlot(confusionMatrix):
    for i in range(len(confusionMatrix)):
        fig, ax = plt.subplots(figsize=(3, 3))
        ax = sns.heatmap(confusionMatrix[i][5], annot=True, cbar=False)
        plt.xlabel("True label")
        plt.ylabel("Predicted label")
        plt.show()

# Importing samples from Excel files and treating data.
trainingSamples = pd.read_excel("amostratreinamento.xlsx")
testSamples = pd.read_excel("amostrateste.xlsx")

X = trainingSamples.iloc[:, 1:-1]
y = trainingSamples.iloc[:-1]

X_test = testSamples.iloc[:, 1:-1]
y_test = testSamples.iloc[:, -1]

# Data classification (Model, X_validation, y_validation, X_train, y_train,
# all y predicted, respective data set names, evaluation results).
trainedModel, y_validation, X_train, y_train, dataPrediction, dataSet, y_trainEvaluation, \
y_validationEvaluation, y_testEvaluation = dataClassification(X, y, X_test, y_test)
evaluationList = [y_trainEvaluation, y_validationEvaluation, y_testEvaluation]

# Data classification plots (Confusion matrix and ROC curve [in progress]).
confusionMatrixPlot(evaluationList)

# Saving prediction data into an Excel worksheet.
saveDataToExcel(X_train, y_train, y_validation, y_test, dataPrediction)

# Saving evaluation data into an Excel worksheet.
saveEvaluationToExcel(evaluationList, dataSet)

# Save as pickle file.
serializingRnaStructure('ClassificationSample', trainedModel)
