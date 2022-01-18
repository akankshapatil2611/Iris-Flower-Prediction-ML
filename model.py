from re import X
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
import pickle as pk

# Load csv file
df = pd.read_csv("Iris.csv")

# Top 5 rows
print(df.head())

# Label Encoding
df = df.replace({'Species':{'Iris-setosa':0, 'Iris-versicolor':1, 'Iris-virginica':2}})

# Select dependent and independent variables
x = df[['SepalLengthCm', 'SepalWidthCm',  'PetalLengthCm',  'PetalWidthCm']]
y = df['Species']

# Split into train and test
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=0)

# Instantiate model
classifier = DecisionTreeClassifier(criterion='entropy', max_depth=2)

# Fit the model
classifier.fit(X_train, y_train)

#make pickle file of model
pk.dump(classifier, open("i_model.pkl", 'wb'))

p = classifier.predict([[5.1, 3.5, 1.4, 0.2]])
print(p)