from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
from sklearn.ensemble import VotingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np

# Load the data from result.csv
data = np.loadtxt(open('result.csv', 'rb'), delimiter=',')
X = data[:, 0:5]  # Features (0 to 4)
y = data[:, 5]    # Target (5th column)

# Split the data into training and test sets (75% train, 25% test)
x_train, x_test, y_train, y_test = train_test_split(X, y, random_state=0, test_size=0.25)

# Define individual classifiers
classifiers = {
    "KNN": KNeighborsClassifier(n_neighbors=5),
    "Naive Bayes": GaussianNB(),
    "Ensemble (KNN + NB)": VotingClassifier(estimators=[
        ('GaussianNB', GaussianNB()),
        ('KNN', KNeighborsClassifier(n_neighbors=5))
    ], voting='soft')
}

# Initialize lists to store metrics
f1_scores = []
precision_scores = []
recall_scores = []

# Fit each classifier and calculate F1, precision, and recall scores
for name, clf in classifiers.items():
    clf.fit(x_train, y_train)
    predictions = clf.predict(x_test)
    
    # Calculate F1, precision, and recall
    f1 = f1_score(y_test, predictions, average='binary')  # Adjust 'binary' for binary classification
    precision = precision_score(y_test, predictions, average='binary')
    recall = recall_score(y_test, predictions, average='binary')
    
    # Append the scores
    f1_scores.append((name, f1 * 100))
    precision_scores.append((name, precision * 100))
    recall_scores.append((name, recall * 100))

# Sort the scores by F1 score for better visualization
f1_scores.sort(key=lambda x: x[1], reverse=True)
precision_scores.sort(key=lambda x: x[1], reverse=True)
recall_scores.sort(key=lambda x: x[1], reverse=True)

# Extract names and values for each metric
classifier_names = [x[0] for x in f1_scores]
f1_values = [x[1] for x in f1_scores]
precision_values = [x[1] for x in precision_scores]
recall_values = [x[1] for x in recall_scores]

# Plot the metrics in a combined horizontal bar chart
fig, ax = plt.subplots(figsize=(10, 7))

bar_width = 0.2
indices = np.arange(len(classifier_names))

# Plot each metric
ax.barh(indices, f1_values, bar_width, label='F1 Score', color='skyblue')
ax.barh(indices + bar_width, precision_values, bar_width, label='Precision', color='orange')
ax.barh(indices + 2 * bar_width, recall_values, bar_width, label='Recall', color='green')

# Add labels and title
ax.set_xlabel('Scores (%)')
ax.set_yticks(indices + bar_width)
ax.set_yticklabels(classifier_names)
ax.set_title('Comparison of F1 Score, Precision, and Recall for KNN, Naive Bayes, and Ensemble Classifier')
ax.legend()

# Display the scores on the bars
for i, (f1, precision, recall) in enumerate(zip(f1_values, precision_values, recall_values)):
    ax.text(f1 + 1, i, f"{f1:.2f}%", va='center', ha='left', fontweight='bold')
    ax.text(precision + 1, i + bar_width, f"{precision:.2f}%", va='center', ha='left', fontweight='bold')
    ax.text(recall + 1, i + 2 * bar_width, f"{recall:.2f}%", va='center', ha='left', fontweight='bold')

# Show the plot
plt.tight_layout()
plt.show()

