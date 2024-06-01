# import required packages
import pandas as pd
import scipy
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier


print("Grabbing parsed emails...")
# import the parsed_emails_cleaned.csv file
df = pd.read_csv("parsed_emails_cleaned.csv", encoding='latin-1')

# create classifier from xOrigin column
classifier = df['xOrigin']

# clean the classifier: trim the names and turn all into lowercase
classifier = classifier.str.lower()
classifier = classifier.str.replace(' ', '')

print("Grabbing bag of words...")
# import the npz file bag of words
bagOfWords = scipy.sparse.load_npz("bow.npz")

print("Running KNN")


# perform stratified random sampling with 70% training and 30% testing, random_state = 2022
bow_train, bow_test, class_train, class_test = train_test_split(bagOfWords, classifier, test_size=0.3, random_state=2022, stratify= classifier)

# use the training and testing data to train 30 KNN classifiers
# The KNN classifiers should be constructed to use the weights from the array below:
weightArray = ['uniform', 'distance']
# The KNN classifiers should be constructed to use the ks from the array below:
kArray = [5, 50, 100, 200, 550]
# The KNN classifiers should be constructed to use the metrics from the array below:
metricArray = ['euclidean', 'manhattan', 'cosine']

# create a dataframe to store the score/accuracy, the weight type, the k value, and the distance metric
df = pd.DataFrame(columns=['weight', 'k', 'metric', 'accuracy'])

# loop throuh the weightArray, kArray, and metricArray and create a KNN classifier for each combination of hyperparameters
for weight in weightArray:
    for k in kArray:
        for metric in metricArray:
            # create classifier
            knn = KNeighborsClassifier(n_neighbors=k, weights=weight, algorithm='auto', metric=metric, metric_params=None, n_jobs=-1)
            print("Training KNN with weight = " + weight + " , k = " + str(k) + " , and metric = " + metric)
            # fit the classifier
            knn.fit(bow_train, class_train)
            print("Model has been fit")
            # score the classifier
            score = knn.score(bow_test, class_test)
            print("Model has been scored")
            print("KNN with weight = " + weight + " , k = " + str(k) + " , and metric = " + metric + " scored " + str(score))
            # create a temporary dataframe to store the score/accuracy, the weight type, the k value, and the distance metric
            tempDf = pd.DataFrame([[weight, k, metric, score]], columns=['weight', 'k', 'metric', 'accuracy'])

            # append the temporary dataframe to the main dataframe using concat
            df = pd.concat([df, tempDf])
            print("Model has been added to dataframe")

# store the final dataframe as a csv and a pickle for redundancy and accessibility
df.to_csv("KNN.csv", index=False)
df.to_pickle("KNN.pkl")

# print status update
print("KNN has been run and the results have been stored in KNN.csv and KNN.pkl")
print("The final dataframe is:")

# print the final dataframe for easy viewing
print(df)

# print done
print("Done")