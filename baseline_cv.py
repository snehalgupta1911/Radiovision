import numpy as np
import common as C
import dataset as D

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.metrics import classification_report, confusion_matrix

recs = D.load_recordings("dataset", 1.0, 0.25, verbose=False)

X, y, sessions = [], [], []

for r in recs:
    if r.session == "session5":
        continue

    g = r.G[r.valid].astype(np.float32)
    if len(g) == 0:
        continue

    X.append(np.concatenate([g.mean(axis=0), g.std(axis=0)]))
    y.append(r.label)
    sessions.append(r.session)

X = np.asarray(X)
y = np.asarray(y)
sessions = np.asarray(sessions)

print("Recordings:", len(y))
print("Sessions included:", sorted(set(sessions)))
print("Features per recording:", X.shape[1])
print("\nRecordings per class:")
for i, name in enumerate(C.CLASSES):
    print(name, int(np.sum(y == i)))

# Each recording appears in exactly one test fold.
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

model = make_pipeline(
    SimpleImputer(strategy="median"),
    StandardScaler(),
    LogisticRegression(max_iter=3000, class_weight="balanced")
)

pred = cross_val_predict(model, X, y, cv=cv)

print("\nWithin-sessions 1–4 cross-validation results")
print("Accuracy:", round(accuracy_score(y, pred) * 100, 2), "%")
print("Balanced accuracy:", round(balanced_accuracy_score(y, pred) * 100, 2), "%")
print("Macro-F1:", round(f1_score(y, pred, average="macro", zero_division=0) * 100, 2), "%")

print("\nPer-class report:")
print(classification_report(
    y, pred,
    labels=np.arange(len(C.CLASSES)),
    target_names=C.CLASSES,
    zero_division=0
))

print("Confusion matrix (rows = true, columns = predicted):")
print(confusion_matrix(y, pred, labels=np.arange(len(C.CLASSES))))
