from pandas import read_csv
dataset = read_csv('C:/Users/310297830/Desktop/data set/pima-indians-diabetes.data.csv', header=None)
print("The raw data is described")
print(dataset.describe())
