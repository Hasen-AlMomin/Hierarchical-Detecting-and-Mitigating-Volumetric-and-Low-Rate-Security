from __future__ import division
import numpy
import os
from sklearn import svm
from collections import deque
import matplotlib.pyplot as plt
from mlxtend.plotting import plot_decision_regions
import numpy as np
from sklearn import svm, datasets
from sklearn.metrics import accuracy_score 
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import VotingClassifier  # Added this import
import time

data = np.loadtxt(open('result.csv', 'rb'), delimiter=',')



FER = 0
SIGS = 1
FSR = 2
PLV = 3
BLV = 4




#Graph1 FER & SIGS
X = data[:, [FER,SIGS]]
y = data[:, 5]
#clf = svm.SVC()
clf = RandomForestClassifier()
clf.fit(X, y)
# Plot Decision Region using mlxtend's awesome plotting function
fig = plt.figure(figsize=(10,8))
fig = plot_decision_regions(X=X, 
                      y=y.astype(int),
                      clf=clf, 
                      legend=2)
plt.title('Random Forest DDoS - Decision Region Boundary', size=16)
plt.xlabel('Speed of Flow Entry')
plt.ylabel('Speed of Source IP')
plt.savefig("Random Forest_graph1.png")




#Graph1 PLV & blv
X = data[:, [PLV,BLV]]
y = data[:, 5]
#clf = svm.SVC()
clf = RandomForestClassifier()
clf.fit(X, y)
# Plot Decision Region using mlxtend's awesome plotting function
fig = plt.figure(figsize=(10,8))
fig = plot_decision_regions(X=X, 
                      y=y.astype(int),
                      clf=clf, 
                      legend=2)
plt.title('Random Forest DDoS - Decision Region Boundary', size=16)
plt.xlabel('stddev flow packets')
plt.ylabel('stddev flow bytes')
plt.savefig("Random Forest_graph2.png")





#Graph3 FER & PLV
X = data[:, [FER,PLV]]
y = data[:, 5]
#clf = svm.SVC()
clf = RandomForestClassifier()

clf.fit(X, y)
# Plot Decision Region using mlxtend's awesome plotting function
fig = plt.figure(figsize=(10,8))
fig = plot_decision_regions(X=X, 
                      y=y.astype(int),
                      clf=clf, 
                      legend=2)
plt.title('Random Forest DDoS - Decision Region Boundary', size=16)
plt.xlabel('speed of flow entry')
plt.ylabel('stddev flow packets')
plt.savefig("Random Forest_graph3.png")






#Graph4 SIGS & FSR
X = data[:, [SIGS,FSR]]
y = data[:, 5]
#clf = svm.SVC()
clf = RandomForestClassifier()
clf.fit(X, y)
# Plot Decision Region using mlxtend's awesome plotting function
fig = plt.figure(figsize=(10,8))
fig = plot_decision_regions(X=X, 
                      y=y.astype(int),
                      clf=clf, 
                      legend=2)
plt.title('Random Forest DDoS - Decision Region Boundary', size=16)
plt.xlabel('speed of src ip')
plt.ylabel('FSR')
plt.savefig("Random Forest_graph4.png")



#Graph4 SIGS & FSR
X = data[:, [FER,FSR]]
y = data[:, 5]
#clf = svm.SVC()
clf = RandomForestClassifier()
clf.fit(X, y)
# Plot Decision Region using mlxtend's awesome plotting function
fig = plt.figure(figsize=(10,8))
fig = plot_decision_regions(X=X, 
                      y=y.astype(int),
                      clf=clf, 
                      legend=2)
plt.title('Random Forest DDoS - Decision Region Boundary', size=16)
plt.xlabel('FER')
plt.ylabel('FSR')
plt.savefig("Random Forest_graph5.png")

