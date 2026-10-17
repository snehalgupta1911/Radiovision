import numpy as np
import common as C
import dataset as D

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.metrics import f1_score, classification_report, confusion_matrix

# Load recordings with the same preprocessing as the neural network
recs = D.load_recordings("dataset", 1.0, 0.25, verbose=False)

X = []
y = []
sessions = []

# One feature vector per recording: mean + standard deviation
for r in recs:
    g = r.G[r.valid].astype(np.float32)

    if len(g) == 0:
        continue

    features = np.concatenate([
        np.mean(g, axis=0),
        np.std(g, axis=0)
    ])

    X.append(features)
    y.append(r.label)
    sessions.append(r.session)

X = np.asarray(X)
y = np.asarray(y)
sessions = np.asarray(sessions)

train = np.isin(sessions, ["session1", "session2", "session3", "session4"])
test = sessions == "session5"

if not train.any() or not test.any():
    raise SystemExit("Could not find both training sessions 1–4 and test session 5.")

print("Training recordings:", int(train.sum()))
print("Test recordings:", int(test.sum()))
print("Feature dimensions:", X.shape[1])

model = make_pipeline(
    SimpleImputer(strategy="median"),
    StandardScaler(),
    LogisticRegression(max_iter=3000, class_weight="balanced")
)

model.fit(X[train], y[train])
pred = model.predict(X[test])

print("\nSession 5 results")
print("Accuracy:", round(accuracy_score(y[test], pred) * 100, 2), "%")
print("Balanced accuracy:", round(balanced_accuracy_score(y[test], pred) * 100, 2), "%")
print("Macro-F1:", round(f1_score(y[test], pred, average="macro", zero_division=0) * 100, 2), "%")

print("\nPer-class report:")
print(classification_report(
    y[test], pred,
    labels=np.arange(len(C.CLASSES)),
    target_names=C.CLASSES,
    zero_division=0
))

print("Confusion matrix (rows = true, columns = predicted):")
print(confusion_matrix(y[test], pred, labels=np.arange(len(C.CLASSES))))
