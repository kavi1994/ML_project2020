# -----LOADING AND CLEANING THE DATA--------#

# Importing the libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

#%matplotlib inline

#Ignore warnings
import warnings
warnings.filterwarnings("ignore")

# read data file and put data into object
df = pd.read_csv('adult_dataset.csv')

# get the types of each column data
df.info()

# visualize data 
df.head()

# get the missing values of each row, missing values represent as '?'
df_1 = df[df.workclass == '?']
df_1

# remove missng values workclass, occupation and native country columns
df = df[df['workclass'] != '?']
df = df[df['occupation'] != '?']
df = df[df['native.country'] != '?']
df.head()

# clean dataframe fetures data types
df.info()

# -----DATA PREPARATION--------#

from sklearn import preprocessing


# encode categorical variables using Label Encoder

# select all categorical variables
df_categorical = df.select_dtypes(include=['object'])
df_categorical.head()

# apply label endcorder to df_categorical variable
le = preprocessing.LabelEncoder()
df_categorical = df_categorical.apply(le.fit_transform)
df_categorical.head()

# concat df_categorical with original df 
df = df.drop(df_categorical.columns, axis=1)
df = pd.concat([df, df_categorical], axis=1)
df.head()

# check column types after concatenate 
df.info()

# convert target variable 'income' to catagorical
df['income'] = df['income'].astype('category')
 
 
 # -----MODEL BUILDING AND EVALUTION--------#

# importing train-test-split library
from sklearn.model_selection import train_test_split

# putting feature variable to X
X = df.drop('income',axis=1)

# Putting response variable to y
y = df['income']

# splitting data into train and test 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state = 99)


# Importing decision tree classifier from sklearn library
from sklearn.tree import DecisionTreeClassifier

# Fitting the decision tree with hyperparameters
# max_depth which is 5 so that we can plot and read the tree.
dt_default = DecisionTreeClassifier(criterion="entropy", 
                                    random_state=100,
                                    max_depth=3, 
                                    min_samples_leaf=3)
dt_default.fit(X_train, y_train)

# Importing classification report and confusion matrix from sklearn metrics
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Making predictions
y_pred_default = dt_default.predict(X_test)

# Printing classification report
print("Classification Report : ")
print(classification_report(y_test, y_pred_default))

# Printing confusion matrix and accuracy

from sklearn.metrics import confusion_matrix
import seaborn as sns; sns.set()
#get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt


print("Confusion Matrix : ")
print(confusion_matrix(y_test,y_pred_default))

mat = confusion_matrix(y_test, y_pred_default)
sns.heatmap(mat.T, square=True, annot=True, fmt='d', cbar=False)
plt.xlabel('true label')
plt.ylabel('predicted label');

print("Accuracy : " , accuracy_score(y_test,y_pred_default) * 100)

# -----DRAW THE DECISION TREE---------#

# Importing required packages for visualization
from IPython.display import Image  
from sklearn.externals.six import StringIO  
from sklearn.tree import export_graphviz
import pydotplus, graphviz

# Putting features
features = list(df.columns[1:])
features

dot_data = StringIO()  
export_graphviz(dt_default, out_file=dot_data,feature_names=features, filled=True,rounded=True)

graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
graph.write_png('predict_salary.png')
Image(graph.create_png())

# get max_depth=8, min_samples_leaf=8 decision tree 
dt_default = DecisionTreeClassifier(criterion="entropy", 
                                    random_state=100,
                                    max_depth=8,
                                    min_samples_leaf=8)
dt_default.fit(X_train, y_train)


dot_data = StringIO()  
export_graphviz(dt_default, out_file=dot_data,feature_names=features, filled=True,rounded=True)

graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
graph.write_png('predict_salary_maxdepth_8.png')
Image(graph.create_png())
 
 pickle.dump(regressor, open('model.pkl','wb'))

model = pickle.load(open('model.pkl','rb'))
print(model.predict([[41, Private, 264663, Some-college, Separated, Prof-speciality, own-child, White, Female, 0, 3900, 40, United States ]]))
 
 
 