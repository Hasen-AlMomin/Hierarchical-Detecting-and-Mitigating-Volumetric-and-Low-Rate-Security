from sklearn import svm, datasets
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score 
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
import time

# Load the data in numpy array
data = np.loadtxt(open('result.csv', 'rb'), delimiter=',')
X = data[:, 0:5]
y = data[:, 5]

#test-size = 1- (test size=0.1)= 0.9 which will take 90%  of the data
# Split the data into training & test data. Test-size is 0.25 (25%) of the data
x_train, x_test, y_train, y_test = train_test_split(X, y, random_state=0, test_size=0.25)

# Define the classifiers
classifiers = [
    #svm.SVC(),
    #MLPClassifier(),
    MLPClassifier(solver='sgd', alpha=1e-5,hidden_layer_sizes=(2, 3, 2), random_state=1),
    KNeighborsClassifier(),
    RandomForestClassifier(),
    GaussianNB(),
    tree.DecisionTreeClassifier()
]

# Calculate accuracy scores for each classifier
accuracy_scores = []
learning_times = []

for clf in classifiers:
    start_time = time.time()  # Start time measurement
    clf.fit(x_train, y_train)
    end_time = time.time()  # End time measurement
    
    predictions = clf.predict(x_test)
    accuracy = accuracy_score(y_test, predictions)
    accuracy_scores.append(accuracy)
    
    learning_time = end_time - start_time
    learning_times.append(learning_time)
    

# Sort the accuracy scores in descending order
#sorted_scores = sorted(zip(classifiers, accuracy_scores), key=lambda x: x[1], reverse=True)

# Sort the accuracy scores and learning times in descending order based on accuracy
sorted_scores = sorted(zip(classifiers, accuracy_scores, learning_times),
                       key=lambda x: x[1], reverse=True)

# Print the accuracy scores in descending order
#for clf, accuracy in sorted_scores:
#    clf_name = clf.__class__.__name__
#    print(clf_name + " Accuracy:", accuracy)
#    print(clf_name + " learning Time:", accuracy)

# Print the accuracy scores and learning times in descending order
for clf, accuracy, time_taken in sorted_scores:
    clf_name = clf.__class__.__name__
    accuracy_percentage = accuracy * 100
    print(clf_name + " Accuracy: {:.2f}%".format(accuracy_percentage))
    print(clf_name + " Learning Time:", time_taken)
