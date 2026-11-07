from sklearn import tree
from sklearn.metrics import precision_score, recall_score, precision_recall_curve, accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
import time

# Load the data from result.csv
data = np.loadtxt(open('result.csv', 'rb'), delimiter=',')
X = data[:, 0:5]  # Features (0 to 4)
y = data[:, 5]    # Target (5th column)

# Split the data into training and test sets (75% train, 25% test)
x_train, x_test, y_train, y_test = train_test_split(X, y, random_state=0, test_size=0.25)

# Define the combined classifier using VotingClassifier with soft voting
combined_classifier = VotingClassifier(estimators=[
    ('GaussianNB', GaussianNB()),
    ('KNN', KNeighborsClassifier(n_neighbors=5))
], voting='soft')

# Fit the combined classifier
start_time = time.time()
combined_classifier.fit(x_train, y_train)
end_time = time.time()

# Make predictions
predictions = combined_classifier.predict(x_test)
accuracy = accuracy_score(y_test, predictions)
learning_time = end_time - start_time

# Print the accuracy and learning time
print(f"Combined Classifier (Gaussian Naive Bayes and KNN) Accuracy: {accuracy * 100:.2f}%")
print(f"Combined Classifier (Gaussian Naive Bayes and KNN) Learning Time: {learning_time:.4f} seconds")

# Calculate precision and recall for the combined classifier
precision = precision_score(y_test, predictions, average='binary')
recall = recall_score(y_test, predictions, average='binary')

print(f"Combined Classifier Precision: {precision * 100:.2f}%")
print(f"Combined Classifier Recall: {recall * 100:.2f}%")

# Calculate Confusion Matrix
cm = confusion_matrix(y_test, predictions)

# Plot Confusion Matrix
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(cmap=plt.cm.Blues)
plt.title('Confusion Matrix for Hybrid Voting Classifier (KNN + NB)')
plt.show()

