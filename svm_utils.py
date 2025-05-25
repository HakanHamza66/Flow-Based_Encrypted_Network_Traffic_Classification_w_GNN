import os
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score

def load_meta_features(csv_folder, label_type="appName"):
    X_all = []
    y_all = []

    for file in os.listdir(csv_folder):
        if file.endswith(".csv"):
            df = pd.read_csv(os.path.join(csv_folder, file))

            # Hedef etiket (y)
            if label_type not in df.columns:
                continue
            y = df[label_type]
            X = df.drop(columns=["appName", "category"], errors="ignore")
            X_all.append(X)
            y_all.extend(y)
    X_all = pd.concat(X_all).fillna(0).astype('float32')
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X_all)

    return X_scaled, y_all

def train_svm(X, y, label_name="appName"):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    clf = SVC(kernel='rbf', C=1.0, gamma='scale')
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    print(f"\n🔍 SVM Classification Results ({label_name}):\n")
    print(classification_report(y_test, y_pred))
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average='macro', zero_division=0)
    rec = recall_score(y_test, y_pred, average='macro', zero_division=0)
    f1 = f1_score(y_test, y_pred, average='macro', zero_division=0)

    return {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1_score": f1
    }
