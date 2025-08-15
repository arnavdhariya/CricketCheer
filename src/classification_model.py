import numpy as np
import os
from analyze_audio import analyze_audio
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score

nonnp_X = []
nonnp_y = []

for filename in os.listdir("../data/goodshot"):
    nonnp_X.append(analyze_audio("../data/goodshot/" + filename))
    nonnp_y.append(1)
for filename in os.listdir("../data/badshot"):
    nonnp_X.append(analyze_audio("../data/badshot/" + filename))
    nonnp_y.append(0)

X = np.array(nonnp_X)
y = np.array(nonnp_y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
model = RandomForestClassifier(n_estimators=100)
model.fit(X_train, y_train)

y_pred = model.predict(X_train)
y_prob = model.predict_proba(X_train)

# Accuracy
accuracy = accuracy_score(y_pred, y_train)
print(f"Test Accuracy:", accuracy)
#
# features = analyze_audio("../data/test2.wav")
#
# # Ensure features are 2D (scikit-learn expects shape [n_samples, n_features])
# features = np.array(features).reshape(1, -1)
#
# # Predict
# prediction = model.predict(features)
# print(prediction)
# proba = model.predict_proba(features)
# print(proba)
# print("Detailed Report:")
# print("Index\tPredicted\tProbability(Good Shot)\tActual\tCorrect")
# for i, (pred, prob, actual) in enumerate(zip(y_pred, y_prob[:, 1], y_train), 1):
#     correct = "Yes" if pred == actual else "No"
#     label = "Good" if pred == 1 else "Bad"
#     actual_label = "Good" if actual == 1 else "Bad"
#     print(f"{i}\t{label}\t{prob:.2f}\t{actual_label}\t{correct}")