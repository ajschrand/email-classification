import pandas
import matplotlib.pyplot as plt
import colorsys
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# load the data
features = pandas.read_csv('features.csv')
print("Features loaded")

# separate the features from the labels
X = features.drop(['xOrigin'], axis=1)
Y = features['xOrigin']

# scale the features
scaler = StandardScaler()
X = scaler.fit_transform(X)

totalVariances = []
for i in range(1, len(X[0])):
    print(f'Number of components: {i}')
    pca = PCA(n_components=i)
    X_pca = pca.fit_transform(X)
    sumVariance = sum(pca.explained_variance_ratio_)
    totalVariances.append(sumVariance)

# plot the cumulative variance
plt.plot(totalVariances)
plt.title('Total Variance by Number of Components')
plt.xlabel('Number of Components')
plt.ylabel('Total Variance')
plt.show()

pca = PCA(n_components=21)
X = pca.fit_transform(X)

# print the amount of variance explained by each component
print(pca.explained_variance_ratio_)

# print the cumulative variance explained by the components
print(sum(pca.explained_variance_ratio_))

# for each component, print the name of the feature with the highest variance
for i in range(21):
    print(f'Component {i + 1}: {features.columns[1:][pca.components_[i].argmax()]}')

# create a color for each author
N = len(Y.unique())
HSV_tuples = [(x*1.0/N, 0.5, 0.5) for x in range(N)]
RGB_tuples = map(lambda x: colorsys.hsv_to_rgb(*x), HSV_tuples)
colors = list(map(lambda x: '#%02x%02x%02x' % (int(x[0]*255), int(x[1]*255), int(x[2]*255)), RGB_tuples))

# map each author to an index
author_to_index = {}
for index, author in enumerate(Y.unique()):
    author_to_index[author] = index

# create a list of colors for each author
colors_for_each_author = []
for author in Y:
    colors_for_each_author.append(colors[author_to_index[author]])
    
# plot the data
plt.scatter(X[:, 0], X[:, 1], c=colors_for_each_author, alpha=0.5)

# add axis labels
plt.xlabel('First Principal Component')
plt.ylabel('Second Principal Component')

# show the plot
plt.show()