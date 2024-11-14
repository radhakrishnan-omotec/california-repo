# Import libraries and tools
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
# Here, we’re telling Python to ignore any warning messages. Warnings won’t stop the code from running, 
# but they can clutter the screen. This is like putting on noise-canceling headphones to block out 
# distractions so we can focus on the main task.
warnings.filterwarnings("ignore")

# Assuming X and y are already defined and preprocessed
# Split the data into training and testing sets
# We’re splitting our data into two groups: one for learning (training) and one for testing. 
# We keep 80% for practice and 20% as a quiz to check if our models learned well. 
# Setting random_state=42 ensures the split happens the same way every time, 
# like using the same study guide each time you review.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize features
# Here, we’re using a tool called a “scaler” to make our data easier for models to understand. 
# Some values in our data might be huge (like 1000), and some might be small (like 0.1), 
# which could confuse our models. The scaler shrinks or stretches the numbers so they all 
# fit within a smaller, similar range, making it easier for our models to learn.
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Define a dictionary of models
# We’re creating a dictionary that holds different types of “thinking models” or algorithms. 
# Each of these models is like a different approach to solving the same puzzle. Some models 
# make decisions based on nearest neighbors, others look at random forests of decision trees, 
# and one is an artificial neural network we build layer by layer. The neural network has 
# layers of “neurons” that learn from patterns, and includes “dropout” to prevent memorizing 
# too much, helping it generalize better.
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
# Here, we’re taking each model out of the dictionary, teaching it using the training data, 
# and testing it to see how well it learned. This is like giving each model a mini-exam. 
# If the model is our artificial neural network (the “brain” model), we have to use a special 
# language to train and test it. Otherwise, we use a simpler method. For each model, we calculate 
# the “accuracy” score, which tells us how many answers it got right on the quiz. We then store 
# this score in the results dictionary with the model’s name.
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
# Finally, we’re printing out the scores for each model, which lets us see which model did the 
# best job. This is like showing the grades each model got on the test. By looking at these scores, 
# we can decide which model might be best for our project based on its accuracy.
print("Accuracy Scores:")
for name, acc in results.items():
    print(f"{name}: {acc}")
