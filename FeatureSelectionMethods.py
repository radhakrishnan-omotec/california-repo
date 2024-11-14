import pandas as pd
from sklearn.feature_selection import SelectKBest, mutual_info_classif, f_classif, RFE, SelectFromModel
from sklearn.linear_model import LogisticRegression  # Example classifier for RFE
from sklearn.ensemble import RandomForestClassifier  # Example classifier for SelectFromModel
from sklearn.model_selection import train_test_split

# This code imports necessary tools to help find the best pieces of information (features) in our data.
# "Pandas" is used to manage data easily, and "SelectKBest" helps us choose only the most important features.
# "mutual_info_classif" and "f_classif" are methods to score how useful each feature might be.
# "RFE" (Recursive Feature Elimination) and "SelectFromModel" are other techniques to help us pick out important features.
# Finally, we import Logistic Regression and Random Forest classifiers, which help in ranking the features.

# Assuming X is your feature matrix and y is your target variable
# Replace these with your actual data
# X and y are placeholders here - think of X as the table of data (with many columns), and y as the answer column we're trying to predict.
# You would need to use real data here for the code to work, as this is just a sample.

# Calculate correlation coefficients between features and target
correlations = X.corrwith(y)
# Here, we're calculating something called "correlation" to see how much each feature (column in X) is related to y (the answer column).
# This means we’re checking how closely each feature and the target are linked, with the idea that more related features may be useful predictors.

# Select features with high correlation coefficients
high_corr_features = correlations[abs(correlations) > 0.3].index  # Adjust correlation threshold as needed
# This part checks if any of the features have a strong enough relationship (above 0.3) with the target and selects those features.
# If a feature’s correlation is high, it might mean it has useful information for making predictions, so we save it.

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# We split our data into two parts: training and testing. 
# Training data (80%) is used to teach the model, and testing data (20%) is to see how well the model has learned.
# Setting a “random_state” value makes sure the split is the same every time we run the code, making testing easier.

# Define the number of features to select for SelectKBest and RFE
k_best_features = 10  # Adjust as needed
rfe_num_features = 15  # Adjust as needed
# We decide on the number of features to keep for each method here.
# We’re choosing the top 10 for SelectKBest and the top 15 for RFE (Recursive Feature Elimination), but these numbers can be changed if needed.

# Step 1: Apply different feature selection methods
# This step applies various methods to select the best features that could help predict the target.

# Method 1: SelectKBest with mutual information
skb_mi = SelectKBest(score_func=mutual_info_classif, k=k_best_features)
X_skb_mi = skb_mi.fit_transform(X_train, y_train)
selected_features_skb_mi = X.columns[skb_mi.get_support(indices=True)]
# In this method, we use SelectKBest to choose the top k (10 in this case) features.
# It uses “mutual information” to measure how much each feature tells us about the target.
# After finding the best features, it keeps only those selected, saving their names in 'selected_features_skb_mi'.

# Method 2: SelectKBest with ANOVA F-value
skb_f = SelectKBest(score_func=f_classif, k=k_best_features)
X_skb_f = skb_f.fit_transform(X_train, y_train)
selected_features_skb_f = X.columns[skb_f.get_support(indices=True)]
# Here, we use another method with SelectKBest but now with something called ANOVA F-value.
# This method measures if differences in feature values have a meaningful link to the target.
# Just like before, it picks the top k features and keeps only those.

# Method 3: Recursive Feature Elimination (RFE) with Logistic Regression
rfe_lr = RFE(LogisticRegression(), n_features_to_select=rfe_num_features)
X_rfe_lr = rfe_lr.fit_transform(X_train, y_train)
selected_features_rfe_lr = X.columns[rfe_lr.get_support(indices=True)]
# In this method, Recursive Feature Elimination (RFE) starts with all features and removes the least helpful one at a time.
# It uses Logistic Regression to decide which features to keep, removing others until only the best 15 are left.
# The final list of selected features is saved.

# Method 4: Lasso-based feature selection
lasso_model = RandomForestClassifier()  # Example classifier for SelectFromModel
lasso_selector = SelectFromModel(lasso_model)
X_lasso = lasso_selector.fit_transform(X_train, y_train)
selected_features_lasso = X.columns[lasso_selector.get_support(indices=True)]
# In the fourth method, we use RandomForest (a set of many decision trees) to decide which features are important.
# SelectFromModel automatically chooses important features based on their scores.
# Once it picks the best, we keep those features in the list.

# Step 2: Intersect the selected features from different methods
selected_features_intersection = set(selected_features_skb_mi) & set(selected_features_skb_f) & \
                                set(selected_features_rfe_lr) & set(selected_features_lasso)
# This step combines the selected features from all methods above, keeping only the ones that were selected by each method.
# By finding the overlap (or intersection) of all four lists, we ensure we’re using the most consistently valuable features.

# Add high correlated features to the selected features
final_selected_features = list(selected_features_intersection.union(high_corr_features))
# Lastly, we add any high-correlation features from the earlier correlation check.
# This step creates a final list that includes both highly-correlated features and the best features chosen by the selection methods.

print("Final Selected Features:", final_selected_features)
# Finally, we print the names of the features we selected after all the steps. These are the features we think will best help in predicting the target.
