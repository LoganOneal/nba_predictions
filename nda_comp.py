import numpy as np # Import Numpy and Pandas packages
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_moons, make_circles, make_classification
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

# Download data straight from utkML repo
import requests, io
data_url = 'https://raw.githubusercontent.com/utkML/NBA_comp/main/games_train.csv'
data_download = requests.get(data_url).content

games = pd.read_csv(io.StringIO(data_download.decode('utf-8')))
games_trim = games.drop(columns = ['GAME_ID', 'HOME_TEAM_ID', 'VISITOR_TEAM_ID', 'SEASON'])

# Get y value out, convert to Numpy arrays:
Y = games_trim['HOME_TEAM_WINS'].to_numpy()
X = games_trim.drop(columns = ['HOME_TEAM_WINS']).to_numpy()


from sklearn.model_selection import train_test_split
Xtrain, Xtest, Ytrain, Ytest = train_test_split(X, Y, test_size = 0.1, random_state = 7074)
# Random state sets a seed, so the function draws the same elements randomly every time we run it

#normalize data
from sklearn.preprocessing import StandardScaler
ss = StandardScaler()
Xtrain_scaled = ss.fit_transform(Xtrain) #Only scale the X datasets (not Y)
Xtest_scaled = ss.fit_transform(Xtest)

h = .02  # step size in the mesh
x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                        np.arange(y_min, y_max, h))

#try all models
models = ["Nearest Neighbors", "Linear SVM", "RBF SVM", "Gaussian Process",
         "Decision Tree", "Random Forest", "Neural Net", "AdaBoost",
         "Naive Bayes", "QDA"]

classifiers = [
    KNeighborsClassifier(20),
    SVC(kernel="linear", C=0.025),
    SVC(gamma=2, C=1),
    GaussianProcessClassifier(1.0 * RBF(1.0)),
    DecisionTreeClassifier(max_depth=5),
    RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
    MLPClassifier(alpha=1, max_iter=1000),
    AdaBoostClassifier(),
    GaussianNB(),
    QuadraticDiscriminantAnalysis()]

rng = np.random.RandomState(2)
X += 2 * rng.uniform(size=X.shape)
linearly_separable = (X, y)

# iterate over classifiers
for name, clf in zip(models, classifiers):
    ax = plt.subplot(len(datasets), len(classifiers) + 1, i)
    clf.fit(Xtrain, Ytrain)
    score = clf.score(Xtest, Ytest)

    # Plot the decision boundary. For that, we will assign a color to each
    # point in the mesh [x_min, x_max]x[y_min, y_max].
    if hasattr(clf, "decision_function"):
        Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
    else:
        Z = clf.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]

    # Put the result into a color plot
    Z = Z.reshape(xx.shape)
    ax.contourf(xx, yy, Z, cmap=cm, alpha=.8)

    # Plot the training points
    ax.scatter(Xtrain[:, 0], Xtrain[:, 1], c=Ytrain, cmap=cm_bright,
                edgecolors='k')
    # Plot the testing points
    ax.scatter(Xtest[:, 0], Xtest[:, 1], c=Ytrain, cmap=cm_bright,
                edgecolors='k', alpha=0.6)

    ax.set_xlim(xx.min(), xx.max())
    ax.set_ylim(yy.min(), yy.max())
    ax.set_xticks(())
    ax.set_yticks(())
    if ds_cnt == 0:
        ax.set_title(name)
    ax.text(xx.max() - .3, yy.min() + .3, ('%.2f' % score).lstrip('0'),
            size=15, horizontalalignment='right')
    i += 1

plt.tight_layout()
plt.show()


"""We see that both models do fairly well, but there is area for improvement! A number of things could be done to improve these models, including hyperparameter tuning (we'll talk about this next time) and even selecting other types of models, such as Random Forest, Logistic Regression, or even a simple neural network. """
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import f1_score

knn_normalized = KNeighborsClassifier(n_neighbors=20) # Set up the initial class instance
knn_normalized.fit(Xtrain_scaled, Ytrain) # "fit" or train the model on your data
print('KNN Accuracy:', knn_normalized.score(Xtest_scaled, Ytest)) # Evaluate the accuracy of your classifier
print('KNN F1-score', f1_score(Ytest, knn_normalized.predict(Xtest_scaled)))

from sklearn.svm import SVC
svm_normalized = SVC()
svm_normalized.fit(Xtrain_scaled, Ytrain) # "fit" or train the model on your data
print('SVM Accuracy', svm_normalized.score(Xtest_scaled, Ytest))
print('SVM F1-score', f1_score(Ytest, svm_normalized.predict(Xtest_scaled)))

# First load testing:
data_url = 'https://raw.githubusercontent.com/utkML/NBA_comp/main/games_test.csv'
data_download = requests.get(data_url).content
games_test = pd.read_csv(io.StringIO(data_download.decode('utf-8')))

# Process in same way as testing:
games_test_trim = games_test.drop(columns = ['GAME_ID', 'HOME_TEAM_ID', 'VISITOR_TEAM_ID', 'SEASON'])
kaggle_test = ss.fit_transform(games_test_trim.to_numpy()) # Scale with standard scaler

"""Note that you have to submit your predictions to Kaggle as a CSV file. In the cell below, I show how we can take our predicted values and generate a CSV file that  """

# Use SVM that we trained earlier:
predicted = svm_normalized.predict(kaggle_test)
prediction_df = pd.DataFrame({'id': range(len(predicted)), 'wins': predicted})
print(prediction_df)

# run this line below to generate a CSV file of your predictions:
#prediction_df.to_csv('my_predictions.csv', index = False)

"""After you run the line to save your predictions, do to the little folder icon on the left, find your file, and download it.

## Extra Challenge: Finding Most Predictive Features
This is a challenge for more advanced competitors who might find the classification task too straight forward. The problem of feature selection is finding a small set of features - as small as possible - in the dataset that still gives us an accurate model. If we can do this, we can reasonably claim that these features are most associated with our target outcome, compared to all variables in the dataset.

Your task: find the feature(s) - ideally 1-3 - that are most indicative of if a team will win or not. Below, I show an example with a feature selection method known as Chi-2 feature selection:
"""

from sklearn.feature_selection import chi2, SelectKBest
selector = SelectKBest(score_func = chi2, k = 4)
selector = selector.fit(Xtrain, Ytrain)
Xnew_train = selector.transform(Xtrain)
Xnew_test = selector.transform(Xtest)

svm_selected = SVC()
svm_selected.fit(Xnew_train, Ytrain)
print('SVM selected:', f1_score(Ytest, svm_selected.predict(Xnew_test)))
feature_mask = selector.get_support()
print('Features selected', games_trim.drop(columns = ['HOME_TEAM_WINS']).columns[feature_mask])

"""Looks like assists and rebounds are some of the most indicative variables of whether a team will win or not - who knew! Notice that we're using the un-normalized data here, so this accuracy is comparable to our accuracy achieved on the original dataset. This means that while only using each team's assists and rebounds, we were able to get a model almost as accurate as using the whole dataset. 

There are definitely more sophisticated feature selection algorithms out there - try to find some!
"""