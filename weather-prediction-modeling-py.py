# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.16.4
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# + [markdown] papermill={"duration": 0.009733, "end_time": "2024-04-05T21:25:44.973928", "exception": false, "start_time": "2024-04-05T21:25:44.964195", "status": "completed"}
# # Step 1: Import necessary libraries

# + papermill={"duration": 2.224294, "end_time": "2024-04-05T21:25:47.207645", "exception": false, "start_time": "2024-04-05T21:25:44.983351", "status": "completed"}
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# + [markdown] papermill={"duration": 0.008802, "end_time": "2024-04-05T21:25:47.226024", "exception": false, "start_time": "2024-04-05T21:25:47.217222", "status": "completed"}
# # Step 2: Load the datasets
#

# + papermill={"duration": 0.173292, "end_time": "2024-04-05T21:25:47.408763", "exception": false, "start_time": "2024-04-05T21:25:47.235471", "status": "completed"}
weather_data = pd.read_csv('/kaggle/input/weather-prediction/weather_prediction_dataset.csv')
bbq_labels = pd.read_csv('/kaggle/input/weather-prediction/weather_prediction_bbq_labels.csv')

# + [markdown] papermill={"duration": 0.008698, "end_time": "2024-04-05T21:25:47.426796", "exception": false, "start_time": "2024-04-05T21:25:47.418098", "status": "completed"}
# # # Step 3:show datasets

# + papermill={"duration": 0.049296, "end_time": "2024-04-05T21:25:47.485015", "exception": false, "start_time": "2024-04-05T21:25:47.435719", "status": "completed"}
weather_data.head()

# + papermill={"duration": 0.030098, "end_time": "2024-04-05T21:25:47.524637", "exception": false, "start_time": "2024-04-05T21:25:47.494539", "status": "completed"}
bbq_labels.head()

# + [markdown] papermill={"duration": 0.009606, "end_time": "2024-04-05T21:25:47.544224", "exception": false, "start_time": "2024-04-05T21:25:47.534618", "status": "completed"}
# # I wanna select OSLO_BBQ_weather as y and related features as x

# + [markdown] papermill={"duration": 0.009576, "end_time": "2024-04-05T21:25:47.564370", "exception": false, "start_time": "2024-04-05T21:25:47.554794", "status": "completed"}
# # Step 4: Merge datasets on the 'DATE' column
#

# + papermill={"duration": 0.043187, "end_time": "2024-04-05T21:25:47.617523", "exception": false, "start_time": "2024-04-05T21:25:47.574336", "status": "completed"}
merged_data = pd.merge(weather_data, bbq_labels, on='DATE')
merged_data .head()

# + [markdown] papermill={"duration": 0.009838, "end_time": "2024-04-05T21:25:47.637577", "exception": false, "start_time": "2024-04-05T21:25:47.627739", "status": "completed"}
# # Step 5:Check for missing values in merged_data
#

# + papermill={"duration": 0.024062, "end_time": "2024-04-05T21:25:47.671685", "exception": false, "start_time": "2024-04-05T21:25:47.647623", "status": "completed"}
# Count missing values in all column using isna() method
missing_counts = merged_data.isna().sum().sum()
missing_counts

# + papermill={"duration": 0.03618, "end_time": "2024-04-05T21:25:47.718162", "exception": false, "start_time": "2024-04-05T21:25:47.681982", "status": "completed"}
merged_data.head()

# + [markdown] papermill={"duration": 0.010415, "end_time": "2024-04-05T21:25:47.739380", "exception": false, "start_time": "2024-04-05T21:25:47.728965", "status": "completed"}
# # Step 6: define X and y

# + papermill={"duration": 0.043736, "end_time": "2024-04-05T21:25:47.793870", "exception": false, "start_time": "2024-04-05T21:25:47.750134", "status": "completed"}
# Filter columns related to Oslo and include the date
oslo_columns = merged_data.filter(like='OSLO')

# Create X with 'DATE', 'MONTH', and Oslo-related columns
X = pd.concat([merged_data[['DATE', 'MONTH']], merged_data[oslo_columns.columns]], axis=1)

# Drop the target variable OSLO_BBQ_weather from X
X.drop('OSLO_BBQ_weather', axis=1, inplace=True)

# Target variable
y = merged_data['OSLO_BBQ_weather']

X


# + papermill={"duration": 0.035921, "end_time": "2024-04-05T21:25:47.841274", "exception": false, "start_time": "2024-04-05T21:25:47.805353", "status": "completed"}
merged_data[oslo_columns.columns]

# + papermill={"duration": 0.021058, "end_time": "2024-04-05T21:25:47.873657", "exception": false, "start_time": "2024-04-05T21:25:47.852599", "status": "completed"}
y

# + papermill={"duration": 0.0223, "end_time": "2024-04-05T21:25:47.907381", "exception": false, "start_time": "2024-04-05T21:25:47.885081", "status": "completed"}
# Convert boolean values to numeric (1 for True, 0 for False)
y = y.astype(int)
y

# + [markdown] papermill={"duration": 0.011177, "end_time": "2024-04-05T21:25:47.930247", "exception": false, "start_time": "2024-04-05T21:25:47.919070", "status": "completed"}
# # Step 6: correlations

# + papermill={"duration": 0.031894, "end_time": "2024-04-05T21:25:47.973530", "exception": false, "start_time": "2024-04-05T21:25:47.941636", "status": "completed"}
import numpy as np

# Calculate Pearson correlation coefficients
correlations = X.corrwith(y)

# Absolute values of correlations for better interpretation
correlations = correlations.abs()

# Sort correlations in descending order
sorted_correlations = correlations.sort_values(ascending=False)

print(sorted_correlations)


# + [markdown] papermill={"duration": 0.01174, "end_time": "2024-04-05T21:25:47.997351", "exception": false, "start_time": "2024-04-05T21:25:47.985611", "status": "completed"}
# # Step 7: Feature Selection 

# + [markdown] papermill={"duration": 0.011788, "end_time": "2024-04-05T21:25:48.021195", "exception": false, "start_time": "2024-04-05T21:25:48.009407", "status": "completed"}
# ### About Correlation
#
# 1-Strong Correlation (Close to ±1):
# When the absolute value of the correlation coefficient is close to 1 (either positive or negative), it indicates a strong linear relationship between the two variables. For example, an absolute correlation coefficient of 0.8 or higher suggests a strong linear association between the variables.
#
# 2-Weak Correlation (Close to 0):
#
# When the absolute value of the correlation coefficient is close to 0, it indicates a weak linear relationship or no linear relationship between the variables. A coefficient near 0 means that changes in one variable are not strongly associated with changes in the other variable.
#

# + papermill={"duration": 0.654531, "end_time": "2024-04-05T21:25:48.687744", "exception": false, "start_time": "2024-04-05T21:25:48.033213", "status": "completed"}
import pandas as pd
from sklearn.feature_selection import SelectKBest, mutual_info_classif, f_classif, RFE, SelectFromModel
from sklearn.linear_model import LogisticRegression  # Example classifier for RFE
from sklearn.ensemble import RandomForestClassifier  # Example classifier for SelectFromModel
from sklearn.model_selection import train_test_split

# Assuming X is your feature matrix and y is your target variable
# Replace these with your actual data

# Calculate correlation coefficients between features and target
correlations = X.corrwith(y)

# Select features with high correlation coefficients
high_corr_features = correlations[abs(correlations) > 0.3].index  # Adjust correlation threshold as needed

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the number of features to select for SelectKBest and RFE
k_best_features = 10  # Adjust as needed
rfe_num_features = 15  # Adjust as needed

# Step 1: Apply different feature selection methods
# Method 1: SelectKBest with mutual information
skb_mi = SelectKBest(score_func=mutual_info_classif, k=k_best_features)
X_skb_mi = skb_mi.fit_transform(X_train, y_train)
selected_features_skb_mi = X.columns[skb_mi.get_support(indices=True)]

# Method 2: SelectKBest with ANOVA F-value
skb_f = SelectKBest(score_func=f_classif, k=k_best_features)
X_skb_f = skb_f.fit_transform(X_train, y_train)
selected_features_skb_f = X.columns[skb_f.get_support(indices=True)]

# Method 3: Recursive Feature Elimination (RFE) with Logistic Regression
rfe_lr = RFE(LogisticRegression(), n_features_to_select=rfe_num_features)
X_rfe_lr = rfe_lr.fit_transform(X_train, y_train)
selected_features_rfe_lr = X.columns[rfe_lr.get_support(indices=True)]

# Method 4: Lasso-based feature selection
lasso_model = RandomForestClassifier()  # Example classifier for SelectFromModel
lasso_selector = SelectFromModel(lasso_model)
X_lasso = lasso_selector.fit_transform(X_train, y_train)
selected_features_lasso = X.columns[lasso_selector.get_support(indices=True)]

# Step 2: Intersect the selected features from different methods
selected_features_intersection = set(selected_features_skb_mi) & set(selected_features_skb_f) & \
                                set(selected_features_rfe_lr) & set(selected_features_lasso)

# Add high correlated features to the selected features
final_selected_features = list(selected_features_intersection.union(high_corr_features))

print("Final Selected Features:", final_selected_features)



# + papermill={"duration": 0.023838, "end_time": "2024-04-05T21:25:48.724486", "exception": false, "start_time": "2024-04-05T21:25:48.700648", "status": "completed"}
high_corr_features

# + papermill={"duration": 0.020855, "end_time": "2024-04-05T21:25:48.757204", "exception": false, "start_time": "2024-04-05T21:25:48.736349", "status": "completed"}
selected_features_skb_mi

# + papermill={"duration": 0.022395, "end_time": "2024-04-05T21:25:48.791725", "exception": false, "start_time": "2024-04-05T21:25:48.769330", "status": "completed"}
selected_features_skb_f

# + papermill={"duration": 0.022649, "end_time": "2024-04-05T21:25:48.826975", "exception": false, "start_time": "2024-04-05T21:25:48.804326", "status": "completed"}
selected_features_rfe_lr

# + papermill={"duration": 0.021665, "end_time": "2024-04-05T21:25:48.916402", "exception": false, "start_time": "2024-04-05T21:25:48.894737", "status": "completed"}
selected_features_lasso

# + papermill={"duration": 0.021117, "end_time": "2024-04-05T21:25:48.950029", "exception": false, "start_time": "2024-04-05T21:25:48.928912", "status": "completed"}
selected_features_intersection

# + papermill={"duration": 0.021284, "end_time": "2024-04-05T21:25:48.983629", "exception": false, "start_time": "2024-04-05T21:25:48.962345", "status": "completed"}
final_selected_features

# + [markdown] papermill={"duration": 0.012194, "end_time": "2024-04-05T21:25:49.008303", "exception": false, "start_time": "2024-04-05T21:25:48.996109", "status": "completed"}
# # Step 8: Define new X and y based on new features

# + papermill={"duration": 0.035064, "end_time": "2024-04-05T21:25:49.055938", "exception": false, "start_time": "2024-04-05T21:25:49.020874", "status": "completed"}
X=X[final_selected_features]
X

# + papermill={"duration": 0.022992, "end_time": "2024-04-05T21:25:49.091918", "exception": false, "start_time": "2024-04-05T21:25:49.068926", "status": "completed"}
y

# + [markdown] papermill={"duration": 0.012608, "end_time": "2024-04-05T21:25:49.117519", "exception": false, "start_time": "2024-04-05T21:25:49.104911", "status": "completed"}
#
# ### Dropout is a regularization technique commonly used in neural networks to prevent overfitting. While dropout can help in improving the generalization ability of the model and reduce overfitting, it is not specifically designed to address multicollinearity issues in the input features.

# + [markdown] papermill={"duration": 0.013665, "end_time": "2024-04-05T21:25:49.145703", "exception": false, "start_time": "2024-04-05T21:25:49.132038", "status": "completed"}
# # Step 9: Train and evaluate models

# + papermill={"duration": 16.549059, "end_time": "2024-04-05T21:26:05.707860", "exception": false, "start_time": "2024-04-05T21:25:49.158801", "status": "completed"}
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
import warnings

# Ignore warnings
warnings.filterwarnings("ignore")

# Assuming X and y are already defined and preprocessed

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Define a dictionary of models
models = {
    'Logistic Regression': LogisticRegression(),
    'Support Vector Machine': SVC(),
    'Random Forest': RandomForestClassifier(),
    'K-Nearest Neighbors': KNeighborsClassifier(),
    'Gaussian Naive Bayes': GaussianNB(),
    'Decision Tree': DecisionTreeClassifier(),
    'Artificial Neural Network': Sequential([
        Dense(units=100, activation='relu', input_shape=(X_train_scaled.shape[1],)),
        Dropout(0.2),  # Add dropout with a dropout rate of 20%
        Dense(units=1, activation='sigmoid')
    ])
}

# Train and evaluate each model
results = {}
for name, model in models.items():
    if isinstance(model, Sequential):  # For TensorFlow models
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        model.fit(X_train_scaled, y_train, epochs=20, batch_size=32, verbose=0)
        accuracy = model.evaluate(X_test_scaled, y_test, verbose=0)[1]
    else:  # For scikit-learn models
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        accuracy = accuracy_score(y_test, y_pred)
    results[name] = accuracy

# Print the results
print("Accuracy Scores:")
for name, acc in results.items():
    print(f"{name}: {acc}")


# + [markdown] papermill={"duration": 0.012833, "end_time": "2024-04-05T21:26:05.734152", "exception": false, "start_time": "2024-04-05T21:26:05.721319", "status": "completed"}
# # Step 10: Compare result
# ## Step 10.1: Accuracy Scores

# + papermill={"duration": 0.607416, "end_time": "2024-04-05T21:26:06.354650", "exception": false, "start_time": "2024-04-05T21:26:05.747234", "status": "completed"}
import matplotlib.pyplot as plt
import seaborn as sns

# Define models and their accuracy scores (replace these with your actual results)
models = ['Logistic Regression', 'Support Vector Machine', 'Random Forest', 'K-Nearest Neighbors', 'Gaussian Naive Bayes', 'Decision Tree', 'Artificial Neural Network']
accuracy_scores = [0.9658, 0.9589, 1.0, 0.9343, 0.9193, 1.0, 0.9767]

# Create a bar plot for model comparison
plt.figure(figsize=(10, 6))
sns.barplot(x=accuracy_scores, y=models, palette='viridis')
plt.xlabel('Accuracy Score')
plt.title('Model Comparison: Accuracy Scores')
plt.show()


# + [markdown] papermill={"duration": 0.013622, "end_time": "2024-04-05T21:26:06.382000", "exception": false, "start_time": "2024-04-05T21:26:06.368378", "status": "completed"}
# # Step 10.2: Correlation Heatmap

# + papermill={"duration": 0.373104, "end_time": "2024-04-05T21:26:06.768679", "exception": false, "start_time": "2024-04-05T21:26:06.395575", "status": "completed"}
import seaborn as sns
import matplotlib.pyplot as plt

# Calculate correlation coefficients between features and target
correlations = X.corrwith(y)

# Create a dataframe for visualization
corr_df = pd.DataFrame({'Features': correlations.index, 'Correlation': correlations.values})
corr_df = corr_df.set_index('Features')

# Plotting the correlation heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(corr_df, annot=True, cmap='coolwarm', linewidths=.5)
plt.title('Correlation Heatmap')
plt.show()


# + [markdown] papermill={"duration": 0.014009, "end_time": "2024-04-05T21:26:06.797220", "exception": false, "start_time": "2024-04-05T21:26:06.783211", "status": "completed"}
# # Step 10.3:feature_importances

# + papermill={"duration": 0.554549, "end_time": "2024-04-05T21:26:07.365993", "exception": false, "start_time": "2024-04-05T21:26:06.811444", "status": "completed"}
import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import RandomForestClassifier

# Initialize and train the Random Forest model
rf_model = RandomForestClassifier()
rf_model.fit(X_train_scaled, y_train)

# Get feature importances from the trained model
importances = rf_model.feature_importances_

# Get feature names
feature_names = X.columns

# Sort feature importances in descending order
indices = np.argsort(importances)[::-1]

# Plot the feature importances
plt.figure(figsize=(10, 6))
plt.title("Feature Importance - Random Forest")
plt.bar(range(X.shape[1]), importances[indices], align='center')
plt.xticks(range(X.shape[1]), feature_names[indices], rotation=90)
plt.xlabel("Feature")
plt.ylabel("Importance")
plt.tight_layout()
plt.show()


# + [markdown] papermill={"duration": 0.014963, "end_time": "2024-04-05T21:26:07.396334", "exception": false, "start_time": "2024-04-05T21:26:07.381371", "status": "completed"}
# # Step 10.4:Confusion Matrix

# + papermill={"duration": 2.737561, "end_time": "2024-04-05T21:26:10.149033", "exception": false, "start_time": "2024-04-05T21:26:07.411472", "status": "completed"}
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

# Initialize models as a dictionary
models = {
    'Logistic Regression': LogisticRegression(),
    'Support Vector Machine': SVC(),
    'Random Forest': RandomForestClassifier(),
    'K-Nearest Neighbors': KNeighborsClassifier(),
    'Gaussian Naive Bayes': GaussianNB(),
    'Decision Tree': DecisionTreeClassifier(),
    'Artificial Neural Network': Sequential([
        Dense(units=100, activation='relu', input_shape=(X_train_scaled.shape[1],)),
        Dropout(0.2),  # Add dropout with a dropout rate of 20%
        Dense(units=1, activation='sigmoid')
    ])
}

# Train each model and calculate confusion matrix
for name, model in models.items():
    if name == 'Artificial Neural Network':
        # Compile the ANN model
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    
    if name == 'Artificial Neural Network':
        y_pred = (y_pred > 0.5)  # Convert probabilities to binary predictions
    
    cm = confusion_matrix(y_test, y_pred)
    
    # Plot confusion matrix
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.title(f'Confusion Matrix - {name}')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.show()


# + [markdown] papermill={"duration": 0.017557, "end_time": "2024-04-05T21:26:10.184753", "exception": false, "start_time": "2024-04-05T21:26:10.167196", "status": "completed"}
# # Step 11: Metric Calculation

# + papermill={"duration": 2.90632, "end_time": "2024-04-05T21:26:13.109820", "exception": false, "start_time": "2024-04-05T21:26:10.203500", "status": "completed"}
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score


# Train each model and calculate confusion matrix
for name, model in models.items():
    if name == 'Artificial Neural Network':
        # Compile the ANN model
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    
    if name == 'Artificial Neural Network':
        y_pred = (y_pred > 0.5)  # Convert probabilities to binary predictions
    
    cm = confusion_matrix(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    # Plot confusion matrix with additional metrics annotations
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=['Predicted Negative', 'Predicted Positive'],
                yticklabels=['Actual Negative', 'Actual Positive'])
    plt.title(f'Confusion Matrix - {name}\nPrecision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f}')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.show()


# + papermill={"duration": 0.027397, "end_time": "2024-04-05T21:26:13.175809", "exception": false, "start_time": "2024-04-05T21:26:13.148412", "status": "completed"}

