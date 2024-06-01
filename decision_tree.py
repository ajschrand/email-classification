import datetime
import matplotlib
import matplotlib.pyplot as plt
import re
import pandas as pd
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Used to remove excess text from the graphical decision tree
def replace_text(obj):
    if type(obj) == matplotlib.text.Annotation:
        txt = obj.get_text()
        txt = re.sub("\nsamples[^$]*class","\nclass",txt)
        obj.set_text(txt)
    return obj

# Read in the data
print("Loading data...")
df = pd.read_csv('features.csv')

# Split data into training and testing
X = df.drop('xOrigin', axis=1)
y = df['xOrigin']

# Use stratified sampling to ensure that the training and testing data have the same distribution of classes
# Random seed is 2022 to be consistent
X_train, X_test, y_train, y_test = train_test_split(
	X, y, test_size=0.3, random_state=2022, stratify=y)

# Make results dataframe
results = pd.DataFrame(columns=['criterion', 'max_depth', 'accuracy', 'time'])

# Create a list of max depths to test (None means no maximum depth)
max_depths = list(range(10, 50, 5)) + [None]

# Create entropy and gini trees for each value of max depth
for idx, i in enumerate(max_depths):
	# Create the trees
	clf_entropy = tree.DecisionTreeClassifier(criterion='entropy', max_depth=i)
	clf_gini = tree.DecisionTreeClassifier(criterion='gini', max_depth=i)

	# Print progress
	print(f"Training decision trees... ({(idx*2 + 1):02}/{len(max_depths)*2})", end="", flush=True)

	# Time the training
	now = datetime.datetime.now()

	# Train the trees
	clf_entropy.fit(X_train, y_train)

	entropy_time = round((datetime.datetime.now() - now).total_seconds(), 2)
	print(f" - {entropy_time}s")

	print(f"Training decision trees... ({(idx*2 + 2):02}/{len(max_depths)*2})", end="", flush=True)
	now = datetime.datetime.now()

	clf_gini.fit(X_train, y_train)

	gini_time = round((datetime.datetime.now() - now).total_seconds(), 2)
	print(f" - {gini_time}s")

	# predict with test set
	y_pred_entropy = clf_entropy.predict(X_test)
	y_pred_gini = clf_gini.predict(X_test)

	# calculate accuracy
	entropy_accuracy = accuracy_score(y_test, y_pred_entropy)
	gini_accuracy = accuracy_score(y_test, y_pred_gini)

	#### This code generates a diagram of the decision tree. It was used in our report.
	# fig, ax = plt.subplots()
	# tree.plot_tree(clf_entropy, ax=ax, fontsize=10, feature_names=X.columns, class_names=clf_entropy.classes_, filled=True)
	# # remove the samples and value labels from the tree
	# [replace_text(obj) for obj in ax.findobj(matplotlib.text.Annotation)]
	# # remove the unused subplot
	# ax.set_title(f"Decision Tree (Entropy) - Accuracy: {entropy_accuracy:.2f}")
	# fig.tight_layout()
	# plt.show()

	results = pd.concat([results, pd.DataFrame([[clf_gini.tree_.max_depth, 'entropy', entropy_accuracy, entropy_time], [clf_gini.tree_.max_depth, 'gini', gini_accuracy, gini_time]], columns=['max_depth', 'criterion', 'accuracy', 'time'])], ignore_index=True)

# Print table of accuracies and times
print(results)

# Change accuracy to a percentage
results['accuracy'] = results['accuracy'] * 100

# Plot of the accuracy of the decision tree for different max_depths and splitting criteria
results.pivot(index='max_depth', columns='criterion', values='accuracy').plot()
for i in results.index:
   plt.plot(results.loc[i, 'max_depth'], results.loc[i, 'accuracy'], 'o', color='orange' if results.loc[i, 'criterion'] == 'gini' else 'blue')
plt.title('Accuracy of Decision Tree Classifiers by Max Depth and Criterion')
plt.xlabel('Max Depth of Tree')
plt.ylabel('Test Accuracy (%)')
plt.show()

# Plot of the time taken to train the decision tree for different max_depths and splitting criteria
results.pivot(index='max_depth', columns='criterion', values='time').plot()
for i in results.index:
   plt.plot(results.loc[i, 'max_depth'], results.loc[i, 'time'], 'o', color='orange' if results.loc[i, 'criterion'] == 'gini' else 'blue')
plt.title('Time to Train Decision Tree Classifiers by Max Depth and Criterion')
plt.xlabel('Max Depth of Tree')
plt.ylabel('Generation Time (s)')
plt.show()