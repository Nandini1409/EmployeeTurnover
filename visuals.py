# Perform necessary imports
from encodings import utf_8
from click import style
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
# %pylab inline
import itertools
# Imports related to classifiers
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_predict
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


# Imports related to confusion matrix
from sklearn.tree import export_graphviz
from IPython.display import Image
import streamlit as st 
st.markdown("""
<style>

    h4:hover{
        color:rgb(255, 75, 75);
};
    
</style>
    """, unsafe_allow_html=True)

original_title = '<marquee> <h4> !!! Welcome To Employee Turnover Prediction System !!! </h4></marquee>'
st.markdown(original_title, unsafe_allow_html=True)

import time
with st.spinner('Wait for it...'):
    time.sleep(5)

# # Function to measure accuracy
def calculate_accuracy(predicted, actual):
	return sum(predicted == actual)/len(actual)

# Function to plot confusion matrix
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, "{}".format(cm[i, j]),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True Class')
    plt.xlabel('Predicted Class')



data = np.genfromtxt("emp_turnover.csv", delimiter=',', skip_header=1,
                     dtype={'names': ('Tenure', 'Age', 'Category', 'Gender', 'Department', 'Dept_Ratio', 'Amount', 'Class'),
                            'formats': ('i4', 'i4', 'i4', 'i4', 'i4', 'f8', 'f8', 'U10')})
# st.subheader("The Number of data Present In dataset is figured below ")
# st.subheader(data.shape)
# st.table(data [:5])

import time
with st.spinner('Wait for it...'):
    time.sleep(1)

da = pd.read_csv('emp_turnover.csv')
sns.pairplot(da.dropna(), vars=['Tenure', 'Age', 'Dept_Ratio', 'Amount'], hue='Class', palette="Set2");
st.image('pairplot.png',caption="Generated pair plot")
# Set the fraction of data which should be in the training set

fraction_training = 0.70

# Function to split training & testing data via the above fraction
def splitdata_train_test(data, fraction_training):
    # shuffle the numpy array
    np.random.seed(0)
    np.random.shuffle(data)
    split = int(len(data)*fraction_training)
    return data[:split], data[split:]

# Function to generate features and targets from data array.
def generate_features_targets(data):
    # assign the last column 'Class' to targets
    targets = data['Class']
    # assign selected columns to features
    features = np.empty(shape=(len(data), 4))
    features[:, 0] = data['Tenure']
    features[:, 1] = data['Age'] 
    features[:, 2] = data['Dept_Ratio'] # feature calculated in csv file via Excel
    features[:, 3] = data['Amount']
        
    return features, targets

# Function to split the data and train a decision tree classifier
def dtc_predict_actual(data):
    # split the data into training and testing sets using a training fraction of 0.7
    train, test = splitdata_train_test(data, fraction_training)

    # generate the feature and targets for the training and test sets
    # i.e. train_features, train_targets, test_features, test_targets
    train_features, train_targets = generate_features_targets(train)
    test_features, test_targets = generate_features_targets(test)

    # instantiate a decision tree classifier
    dtc = DecisionTreeClassifier()

    # train the classifier with the train_features and train_targets
    dtc.fit(train_features, train_targets)

    # get predictions for the test_features
    predictions = dtc.predict(test_features)

    # return the predictions and the test_targets
    return predictions, test_targets


# Call the dtc_predict_actual function and pass data
predicted_class, actual_class = dtc_predict_actual(data)


# calculate the model score using our support function
model_score = accuracy_score(predicted_class, actual_class)
st.write("Our accuracy score:", model_score)

# Print some initial results
st.write("Some initial results...\n   predicted,  actual")
for i in range(10):
    st.write("{}. {}, {}".format(i, predicted_class[i], actual_class[i]))

X, Y = generate_features_targets(data)
# feature extraction
model = ExtraTreesClassifier()
model.fit(X, Y)
st.dataframe(model.feature_importances_)

# split the data into training and testing sets as per above code cell method
train, test = splitdata_train_test(data, fraction_training)

# generate the feature and targets for the training and test sets
train_features, train_targets = generate_features_targets(train)
test_features, test_targets = generate_features_targets(test)

# Create a new decision tree classifier with limited depth and fit
dtree = DecisionTreeClassifier(max_depth=3)
dtree.fit(train_features, train_targets)

from sklearn.metrics import confusion_matrix
from sklearn.metrics import plot_confusion_matrix

# Call function to generate features and targets
features, targets = generate_features_targets(data)

# get predictions using 10-fold cross validation with cross_val_predict
dtc = DecisionTreeClassifier(max_depth=4, criterion='entropy')
predicted = cross_val_predict(dtc, features, targets, cv=10)
st.write(predicted)
# calculate the model score using support function
model_score = accuracy_score(predicted, targets)
st.write("Our accuracy score:", model_score)
# calculate the models confusion matrix using sklearns confusion_matrix function
class_labels = list(set(predicted))
model_cm = confusion_matrix(y_true=targets, y_pred=predicted, labels=class_labels)

# Plot the confusion matrix using the provided functions.

plot_confusion_matrix(dtree,features,targets)
st.title('confusion_matrix')
st.image('dt_confusion_matrix.png',caption="confusion_matrix")

# Get predictions from a random forest classifier

def rf_predict_actual(data, n_estimators):
    # generate the features and targets
    features, targets = generate_features_targets(data)

    # instantiate a random forest classifier
    rfc = RandomForestClassifier(n_estimators=n_estimators)
  
    rfc.fit(features,targets)
    # get predictions using 10-fold cross validation with cross_val_predict
    predicted = cross_val_predict(rfc, features, targets, cv=10)

    # return the predictions and their actual classes
    return rfc,predicted, targets

    
# get the predicted and actual classes
number_estimators = 50    # Number of trees
rfc,predicted, actual = rf_predict_actual(data, number_estimators)

# calculate the model score using the calling the previously created function
accuracy = accuracy_score(predicted, actual)
st.write("Accuracy score:", accuracy)

# calculate the models confusion matrix using sklearns confusion_matrix function
class_labels = list(set(actual))
model_cm = confusion_matrix(y_true=actual, y_pred=predicted, labels=class_labels)

# plot the confusion matrix using the provided functions.
plot_confusion_matrix(rfc,features,targets)
st.title("random forest confusion matrix")
st.image('rf_confusion_matrix.png',caption="random forest confusion matrix")

# Print some initial results, including selected features
st.title("Some initial results...\n")
st.subheader("predicted,  actual, Tenure, Age, Dept_Ratio, Amount")
for i in range(10):
    st.write("{}. {}, {}, {}, {}, {}, {}".format(i, predicted[i], actual[i],
                                          features[i, 0], features[i, 1], features[i, 2], features[i, 3]))

# Create a list, convert to a dataframe and review.

# create empty list
errors = []

# loop through the output of the rf_predict_actual function for the errors of interest
for i in range(len(predicted)):
    
    if (predicted[i] == 'Non-active') & (actual[i] == 'Active'):
        errors.append([predicted[i], actual[i], features[i, 0], features[i, 1], features[i, 2], features[i, 3]])
                
# create a dataframe and set column names
error_df = pd.DataFrame(errors, columns=('predicted', 'actual', 'tenure', 'age', 'dept_ratio', 'amount'))

# take a look
st.write(error_df.head())
st.dataframe(error_df.describe())   # count can be reconciled with the top-right quadrant of the confusion matrix

import pickle
data= {"model":rfc,"tenure":'tenure', "age":'age', "dept_ratio":'dept_ratio', "amount":'amount'}
with open('saved_steps.pkl','wb') as file:
  pickle.dump(data,file)

with open('saved_steps.pkl','rb') as file:
  data = pickle.load(file)

classifier_loaded = data["model"]
# title = st.text_input('Movie title', 'Life of Brian')
tenure = data["tenure"]
age = data["age"]
dept_ratio = data["dept_ratio"]
amount = data["amount"]
