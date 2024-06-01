import pandas
import re
import datetime

def processRow(row):
    # find and generate features in the content of each email
    # Features include:
    # - number of sentences beginning with an uppercase letter
    # - number of sentences beginning with a lowercase letter
    # - number of words
    # - average word length
    # - number of sentences
    # - average sentence length
    # - number of paragraphs
    # - average paragraph length
    # - number of exclamation marks
    # - number of dollar signs
    # - number of ampersands
    # - number of percent signs
    # - number of apostrophes
    # - number of left parentheses
    # - number of right parentheses
    # - number of asterisks
    # - number of plus signs
    # - number of commas
    # - number of hyphens
    # - number of periods
    # - number of slashes
    # - number of colons
    # - number of semicolons
    # - number of pipes
    # - number of less than signs
    # - number of greater than signs
    # - number of equals signs
    # - number of question marks
    # - number of at signs
    # - number of left brackets
    # - number of right brackets
    # - number of backslashes
    # - number of carets
    # - number of underscores
    # - number of accents
    # - number of left braces
    # - number of right braces
    # - number of tilde
    # - number of whitespaces
    # - number of multiple question marks
    # - number of multiple exclamation marks
    # - number of ellipses
    # - number of numbers
    # - number of uppercase words
    # - number of lowercase words
    # - number of appearances of the word "well"
    # - number of appearances of the word "anyhow"
    # - average number of appearances of the word "well"
    # - average number of appearances of the word "anyhow"

    # make a new series to store the features
    features = pandas.Series(dtype='float64')

    # get the content of the email
    content = row['content']

    # split the content into paragraphs
    paragraphs = content.split('\n\n')

    # split the content into sentences
    sentences = re.split(r' *[\.\?!][\'"\)\]]* *', content)

    # split the content into words
    words = re.split(r' ', content)

    # add the xOrigin to the features dataframe
    features['xOrigin'] = row['xOrigin']

    # number of sentences in sentences beginning with a capital letter
    numUppercaseSentences = 0
    # number of sentences beginning with a lowercase letter
    numLowercaseSentences = 0
    for sentence in sentences:
        sentence = sentence.strip()
        if len(sentence) < 1:
            continue

        if sentence[0].isupper():
            numUppercaseSentences += 1
        elif sentence[0].islower():
            numLowercaseSentences += 1

    features['numUppercaseSentences'] = numUppercaseSentences
    features['numLowercaseSentences'] = numLowercaseSentences

    # number of words
    features['numWords'] = len(words)

    # average word length
    features['avgWordLength'] = sum(len(word) for word in words) / len(words)

    # number of sentences
    features['numSentences'] = len(sentences)

    # average sentence length
    features['avgSentenceLength'] = sum(len(sentence) for sentence in sentences) / len(sentences)

    # number of paragraphs
    features['numParagraphs'] = len(paragraphs)

    # average paragraph length
    features['avgParagraphLength'] = sum([len(paragraph) for paragraph in paragraphs]) / len(paragraphs)

    # number of exclamation marks
    features['numExclamationMarks'] = len(re.findall(r'!', content))

    # number of dollar signs
    features['numDollarSigns'] = len(re.findall(r'\$', content))

    # number of ampersands
    features['numAmpersands'] = len(re.findall(r'&', content))

    # number of percent signs
    features['numPercentSigns'] = len(re.findall(r'%', content))

    # number of apostrophes
    features['numApostrophes'] = len(re.findall(r"'", content))

    # number of left parentheses
    features['numLeftParentheses'] = len(re.findall(r'\(', content))

    # number of right parentheses
    features['numRightParentheses'] = len(re.findall(r'\)', content))

    # number of asterisks
    features['numAsterisks'] = len(re.findall(r'\*', content))

    # number of plus signs
    features['numPlusSigns'] = len(re.findall(r'\+', content))

    # number of commas
    features['numCommas'] = len(re.findall(r',', content))

    # number of hyphens
    features['numHyphens'] = len(re.findall(r'-', content))

    # number of periods
    features['numPeriods'] = len(re.findall(r'\.', content))

    # number of slashes
    features['numSlashes'] = len(re.findall(r'/', content))

    # number of colons
    features['numColons'] = len(re.findall(r':', content))

    # number of semicolons
    features['numSemicolons'] = len(re.findall(r';', content))

    # number of pipes
    features['numPipes'] = len(re.findall(r'\|', content))

    # number of less than signs
    features['numLessThanSigns'] = len(re.findall(r'<', content))

    # number of greater than signs
    features['numGreaterThanSigns'] = len(re.findall(r'>', content))

    # number of equals signs
    features['numEqualsSigns'] = len(re.findall(r'=', content))

    # number of question marks
    features['numQuestionMarks'] = len(re.findall(r'\?', content))

    # number of at signs
    features['numAtSigns'] = len(re.findall(r'@', content))

    # number of left brackets
    features['numLeftBrackets'] = len(re.findall(r'\[', content))

    # number of right brackets
    features['numRightBrackets'] = len(re.findall(r'\]', content))

    # number of backslashes
    features['numBackslashes'] = len(re.findall(r'\\', content))

    # number of carets
    features['numCarets'] = len(re.findall(r'\^', content))

    # number of underscores
    features['numUnderscores'] = len(re.findall(r'_', content))

    # number of accents (ASCII 192-255)
    features['numAccents'] = len(re.findall(r'[À-ÿ]', content))

    # number of left braces
    features['numLeftBraces'] = len(re.findall(r'{', content))

    # number of right braces
    features['numRightBraces'] = len(re.findall(r'}', content))

    # number of tilde
    features['numTilde'] = len(re.findall(r'~', content))

    # number of whitespaces
    features['numWhitespaces'] = len(re.findall(r'\s', content))

    # number of multiple question marks
    features['numMultipleQuestionMarks'] = len(re.findall(r'\?\?', content))

    # number of multiple exclamation marks
    features['numMultipleExclamationMarks'] = len(re.findall(r'\!\!', content))

    # number of ellipses
    features['numEllipses'] = len(re.findall(r'\.\.\.', content))

    # number of numbers
    features['numNumbers'] = len(re.findall(r'\d', content))

    # number of uppercase words
    features['numUppercaseWords'] = len(re.findall(r'\b[A-Z]{2,}\b', content))

    # number of lowercase words
    features['numLowercaseWords'] = len(re.findall(r'\b[a-z]{2,}\b', content))

    # number of appearances of the word 'well'
    features['numWell'] = len(re.findall(r'\bwell\b', content))

    # number of appearances of the word 'anyhow'
    features['numAnyhow'] = len(re.findall(r'\banyhow\b', content))

    # average number of appearances of the word 'well'
    features['avgNumWell'] = features['numWell'] / len(words)

    # average number of appearances of the word 'anyhow'
    features['avgNumAnyhow'] = features['numAnyhow'] / len(words)

    return features

# load a dataframe from a pkl file
data = pandas.read_csv('parsed_emails.csv')

# keep only "file", "xOrigin" and "content" columns
data = data[['file', 'xOrigin', 'content']]

# remove rows where content is NaN or empty
data = data.dropna(subset=['content'])
data = data[data['content'] != '']

# get current time
now = datetime.datetime.now()

# loop through each row in the dataframe
results = []
numCompleted = 0
for index, row in data.iterrows():
    # extract features from each email
    results.append(processRow(row))
    numCompleted += 1
    if numCompleted % 10000 == 0:
        print(f'Completed {numCompleted} of {len(data)} rows')

print('Elapsed time:', datetime.datetime.now() - now)

# create a new dataframe from the Series objects in the results list
features = pandas.DataFrame(results)

# remove na rows
features = features.dropna()

# trim and lowercase the author names
features['xOrigin'] = features['xOrigin'].str.strip().str.lower()

# save the dataframe 
features.to_csv('project/features.csv', index=False)