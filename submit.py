import numpy as np # Import Numpy and Pandas packages
import pandas as pd
import requests, io

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_validate
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score


# First load testing:
data_url = 'https://raw.githubusercontent.com/utkML/NBA_comp/main/games_test.csv'
data_download = requests.get(data_url).content
games_test = pd.read_csv(io.StringIO(data_download.decode('utf-8')))

# Download training data straight from utkML repo
data_url = 'https://raw.githubusercontent.com/utkML/NBA_comp/main/games_train.csv'
data_download = requests.get(data_url).content
games = pd.read_csv(io.StringIO(data_download.decode('utf-8')))

# remove unused varaibles
games_trim = games.drop(columns = ['GAME_ID', 'HOME_TEAM_ID', 'VISITOR_TEAM_ID', 'SEASON'])

# remove unused varaibles
ss = StandardScaler()
# Process in same way as testing:
games_test_trim = games_test.drop(columns = ['GAME_ID', 'HOME_TEAM_ID', 'VISITOR_TEAM_ID', 'SEASON'])
kaggle_test = ss.fit_transform(games_test_trim.to_numpy()) # Scale with standard scaler

# balance the dataset (equal wins and losses in Y)
k = games_trim['HOME_TEAM_WINS'].value_counts()[0].min()
def sampling_k_elements(group, k=k):
    if len(group) < k:
        return group
    return group.sample(k)

balanced = games_trim.groupby('HOME_TEAM_WINS').apply(sampling_k_elements).reset_index(drop=True)
balanced['HOME_TEAM_WINS'].value_counts()
print(balanced.columns)

# Get y value out, convert to Numpy arrays:
Y = balanced['HOME_TEAM_WINS'].to_numpy()
X = balanced.drop(columns = ['HOME_TEAM_WINS']).to_numpy()

# generate train test split
Xtrain, Xtest, Ytrain, Ytest = train_test_split(X, Y, test_size = 0.05, random_state = 7074)

Xtrain_scaled = ss.fit_transform(Xtrain) #Only scale the X datasets (not Y)
Xtest_scaled = ss.fit_transform(Xtest)

#train using LogisticRegression
model = LogisticRegression()

model.fit(Xtrain_scaled, Ytrain)

print('Model Accuracy:', model.score(Xtest_scaled, Ytest)) # Evaluate the accuracy of your classifier
print('Model F1-score', f1_score(Ytest, model.predict(Xtest_scaled)))

# Use SVM that we trained earlier:
predicted = model.predict(kaggle_test)
prediction_df = pd.DataFrame({'id': range(len(predicted)), 'wins': predicted})
print(prediction_df)

# run this line below to generate a CSV file of your predictions:
prediction_df.to_csv('my_predictions.csv', index = False)