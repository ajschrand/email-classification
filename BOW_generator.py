# import required packages
import re
import pandas as pd
import numpy as np
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
import scipy
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

# generate english stopwords
stopWords = set(stopwords.words('english'))

# generate the bag of words from the emails in
dataDir = "parsed_emails_cleaned.csv"

# Read csv data into dataframe using pandas
# The first column is the classifier
# The second column is the email body
df = pd.read_csv(dataDir, encoding='latin-1')



# Generate the bag of words for all emails, excluding the author's name from the email contents
# Dictionary of all words in the corpus stored as a list
wordsDictionary = []


print("Cleaning data...")
# Remove the author name from the email body
# Iterate through the emails
for index, row in df.iterrows():
    # if row['xFrom'] is not string, skip it
    if type(row['xFrom']) is not str:
        continue
    
    # Generate the name of the author for exclusion from the dictionary
    nameLineSplit = re.split(r'[\s,]', row['xFrom'])
    lastName = ""
    firstName = ""
    if "," in row['xFrom']:
        lastName = nameLineSplit[0]
        firstName = nameLineSplit[2]
    else:
        lastName = nameLineSplit[1]
        firstName = nameLineSplit[0]
    row['content'] = row['content'].replace(firstName, '')
    row['content'] = row['content'].replace(lastName, '')

    # # progress report: every ten thousand emails, print the index
    # if index % 10000 == 0:
    #     print(index)
    

# Create a new dataframe with the words as the columns
df2 = pd.DataFrame(columns=wordsDictionary)

# use a count vectorizer to generate the bag of words, excluding the author's name from the email contents
print("Generating dictionary...")
vectorizer = CountVectorizer(stop_words=stopWords, token_pattern=r'(?u)\b[a-zA-Z\']+\b')
# fit the vectorizer to the data, generating a dictionary of words
vectorizer.fit(df['content'])

print(len(vectorizer.get_feature_names()))

print("Generating bag of words...")

bagOfWords = vectorizer.transform(df['content'])

# save the bow to a file using npz format and scipy sparse
print("Saving bag of words...")
scipy.sparse.save_npz('bow.npz', bagOfWords, compressed=True)

# save the words dictionary to a file
print("Saving dictionary...")
np.save('dictionary.npy', vectorizer.get_feature_names())