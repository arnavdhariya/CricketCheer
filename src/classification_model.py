import numpy as np
import os
from analyze_audio import analyze_audio
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import joblib

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






