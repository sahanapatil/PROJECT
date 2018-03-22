
from pandas import read_csv
import numpy
from sklearn.preprocessing import Imputer
dataset = read_csv('C:/Users/310297830/Desktop/data set/pima-indians-diabetes.data.csv', header=None)
# mark zero values as missing or NaN
dataset[[1,2,3,4,5]] = dataset[[1,2,3,4,5]].replace(0, numpy.NaN)
# fill missing values with mean column values
dataset.fillna(dataset.mean(), inplace=True)
print("count the number of NaN values in each column")
print(dataset.isnull().sum())
print("printing 20 rows of dataset")
print(dataset.head(20))






# fill missing values with mean column values
values = dataset.values
imputer = Imputer()
transformed_values = imputer.fit_transform(values)
print(" count the number of NaN values in each column")
print(numpy.isnan(transformed_values).sum())


import numpy
from sklearn.preprocessing import Imputer
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
dataset = read_csv('C:/Users/310297830/Desktop/data set/pima-indians-diabetes.data.csv', header=None)
# mark zero values as missing or NaN
dataset[[1,2,3,4,5]] = dataset[[1,2,3,4,5]].replace(0, numpy.NaN)
# split dataset into inputs and outputs
values = dataset.values
X = values[:,0:8]
y = values[:,8]
# fill missing values with mean column values
imputer = Imputer()
transformed_X = imputer.fit_transform(X)
# evaluate an LDA model on the dataset using k-fold cross validation
model = LinearDiscriminantAnalysis()
kfold = KFold(n_splits=3, random_state=7)
result = cross_val_score(model, transformed_X, y, cv=kfold, scoring='accuracy')
print("The accuracy obtained is")
print(result.mean())