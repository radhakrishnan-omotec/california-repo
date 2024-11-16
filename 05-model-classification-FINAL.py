# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.16.4
#   kernelspec:
#     display_name: Python [conda env:dsi] *
#     language: python
#     name: conda-env-dsi-py
# ---



# +
import pandas as pd
import numpy as np
import datetime
import time

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics

from xgboost import XGBRegressor

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.regularizers import l1, l2, l1_l2

# %matplotlib inline
# -

# ### Data
# ---

df = pd.read_csv('./data/combined.csv')

df.head()

df = df.drop(columns=['FIRE_NAME', 'lat', 'long'])

df['bool'] = (df['GIS_ACRES'] > 0) * 1

df['date'] = [x.split('-') for x in df['date']]

df['year'] = [int(x[0]) for x in df['date']]

df['month'] = [int(x[1]) for x in df['date']]

df.shape

df_dummy = pd.get_dummies(df, columns=['month'])

# +
# test = ['2008', '07']
# datetime.datetime(int(test[0]), int(test[1]), 1).timestamp()
# -

df_dummy.head()

df_dummy.info()

df_dummy.head()

# ### Classification
# ---

# +
X = df_dummy.drop(columns=['GIS_ACRES', 'date', 'q_avgtempF', 'q_avghumid', 'q_sumprecip', 'maxtempF', 'mintempF', 'CAUSE', 'bool', 'county'])
y = df_dummy['bool']

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.3)
# -

df['bool'].value_counts(normalize=True)

log_reg = LogisticRegression(max_iter=1000)
log_reg.fit(X_train, y_train)
print(log_reg.score(X_train, y_train))
log_reg.score(X_test, y_test)

pd.set_option('display.max_row', None)

coef_df = pd.DataFrame(log_reg.coef_, columns=X.columns)
coef_df.T.sort_values(by=0, ascending=False).head(77)

log_reg.coef_

sc = StandardScaler()
X_train_sc = sc.fit_transform(X_train)
X_test_sc = sc.transform(X_test)

knn = KNeighborsClassifier(n_neighbors=3, weights='distance')
knn.fit(X_train_sc, y_train)
print(knn.score(X_train_sc, y_train))
knn.score(X_test_sc, y_test)

rf = RandomForestClassifier(n_estimators=150, max_depth=15)
rf.fit(X_train, y_train)
print(rf.score(X_train, y_train))
rf.score(X_test, y_test)

rf_preds = rf.predict(X_test)

rf.feature_importances_

feature_df = pd.DataFrame(rf.feature_importances_, index=X.columns)
feature_df.sort_values(by=0, ascending=False)

log_reg.predict(X_test)
knn.predict(X_test)
rf.predict(X_test)

pred_df = pd.DataFrame(columns=['log_reg', 'knn', 'rf'])
pred_df['log_reg'] = log_reg.predict(X_test)
pred_df['knn'] = knn.predict(X_test)
pred_df['rf'] = rf.predict(X_test)
pred_df['ensemble'] = (pred_df.sum(axis=1)/3).round(0)
pred_df.head()

metrics.accuracy_score(pred_df['ensemble'], y_test)

# +
knn_pipe = Pipeline([
    ('sc', StandardScaler()),
    ('knn', KNeighborsClassifier(n_neighbors=3, weights='distance')),
])

vote = VotingClassifier([
    ('knn_pipe', knn_pipe),
    ('rf', RandomForestClassifier(n_estimators=150, max_depth=15)),
    ('ada', AdaBoostClassifier(n_estimators=150))
])

vote.fit(X_train, y_train)
# -

print(f"Train Score:\t{vote.score(X_train, y_train):.4f}")
print(f"Test Score:\t{vote.score(X_test, y_test):.4f}")

tn, fp, fn, tp = metrics.confusion_matrix(y_test, vote.predict(X_test)).ravel()
metrics.plot_confusion_matrix(vote, X_test, y_test, cmap='Oranges', display_labels=['No fire', 'Fire']);

vote_preds = vote.predict(X_test)

metrics.roc_auc_score(y_test, rf_preds)
print(metrics.recall_score(y_test, rf_preds))
print(metrics.precision_score(y_test, rf_preds))

metrics.roc_auc_score(y_test, vote_preds)
print(metrics.recall_score(y_test, vote_preds))
print(metrics.precision_score(y_test, vote_preds))

metrics.plot_roc_curve(rf, X_test, y_test);





# +
n_input = X_train.shape[1]

model = Sequential()
model.add(BatchNormalization())
model.add(Dense(128, input_shape=(n_input,), activation='relu'))
model.add(Dropout(.05))
model.add(Dense(256, activation='relu'))
model.add(Dropout(.05))
model.add(Dense(64, activation='relu'))
model.add(Dropout(.05))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='bce', optimizer='adam', metrics=['acc', 'Recall', 'Precision'])

# early_stop = EarlyStopping(monitor='val_loss', patience=, verbose=1)

history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=100, batch_size=128, verbose=1)

model.evaluate(X_test, y_test)
# -

history_df = pd.DataFrame(history.history)
history_df.sort_values(by='acc', ascending=False)
