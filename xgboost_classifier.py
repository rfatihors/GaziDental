"""
import os
import json
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
import xgboost as xgb
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

images_dir = 'local/dentalModels/termal_images_split'
labels_dir = 'local/dentalModels/new2_termal_labels_split'

classes = {"inflamation1": 0, "inflamation2": 1, "inflamation3": 2}

def extract_features(image_path, json_path):
    with Image.open(image_path) as img:
        img_array = np.array(img.convert('L'))
        img_array = img_array.flatten()

    with open(json_path, 'r') as f:
        json_data = json.load(f)
        annotations = json_data.get('annotations', [])
        if annotations:
            category_id = annotations[0]['category_id']
        else:
            category_id = 0
    return img_array, category_id

X = []
y = []

for json_filename in os.listdir(labels_dir):
    if json_filename.lower().endswith('.json'):
        json_path = os.path.join(labels_dir, json_filename)
        image_filename = os.path.splitext(json_filename)[0] + '.png'
        image_path = os.path.join(images_dir, image_filename)

        if os.path.exists(image_path):
            img_features, label = extract_features(image_path, json_path)
            X.append(img_features)
            y.append(label)

X = np.array(X)
y = np.array(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = xgb.XGBClassifier(objective='multi:softmax', num_class=len(classes), eval_metric='mlogloss')

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"Doğruluk: %{accuracy * 100:.2f}")

conf_matrix = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=classes.keys(), yticklabels=classes.keys())
plt.xlabel("Tahmin Edilen Sınıf")
plt.ylabel("Gerçek Sınıf")
plt.title("Confusion Matrix")
plt.show()

y_test_binary = np.eye(len(classes))[y_test]
y_pred_proba = model.predict_proba(X_test)

for i, class_name in enumerate(classes.keys()):
    fpr, tpr, _ = roc_curve(y_test_binary[:, i], y_pred_proba[:, i])
    auc = roc_auc_score(y_test_binary[:, i], y_pred_proba[:, i])
    plt.plot(fpr, tpr, label=f'{class_name} (AUC = {auc:.2f})')

plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC-AUC Eğrisi')
plt.legend()
plt.show()

model_file = 'local/best_model.joblib'
joblib.dump(model, model_file)
print(f"Model kaydedildi: {model_file}")

def classify_image(image_path, model):
    img = Image.open(image_path).convert('L')
    img_array = np.array(img).flatten()
    img_array = img_array.reshape(1, -1)

    prediction = model.predict(img_array)
    return prediction[0]

new_image_path = 'local/dentalModels/termal_images_split/termal16_3.png'
predicted_class = classify_image(new_image_path, model)
print(f"Tahmin edilen sınıf: {predicted_class}")
"""



import os
import json
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split, GridSearchCV
import xgboost as xgb
from sklearn.metrics import accuracy_score

images_dir = 'local/dentalModels/termal_images_split'
labels_dir = 'local/dentalModels/new2_termal_labels_split'

classes = {"inflamation1": 0, "inflamation2": 1, "inflamation3": 2}

def extract_features(image_path, json_path):
    with Image.open(image_path) as img:
        img_array = np.array(img.convert('L'))
        img_array = img_array.flatten()

    with open(json_path, 'r') as f:
        json_data = json.load(f)
        annotations = json_data.get('annotations', [])
        if annotations:
            category_id = annotations[0]['category_id']
        else:
            category_id = 0

    return img_array, category_id

X = []
y = []

for json_filename in os.listdir(labels_dir):
    if json_filename.lower().endswith('.json'):
        json_path = os.path.join(labels_dir, json_filename)
        image_filename = os.path.splitext(json_filename)[0] + '.png'
        image_path = os.path.join(images_dir, image_filename)

        if os.path.exists(image_path):
            img_features, label = extract_features(image_path, json_path)
            X.append(img_features)
            y.append(label)

X = np.array(X)
y = np.array(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

xgb_model = xgb.XGBClassifier(objective='multi:softmax', num_class=len(classes), eval_metric='mlogloss')

param_grid = {
    'learning_rate': [0.01, 0.1, 0.2],
    'max_depth': [3, 5, 7],
    'n_estimators': [50, 100, 200],
    'subsample': [0.8, 1.0]
}

grid_search = GridSearchCV(estimator=xgb_model, param_grid=param_grid, cv=3, scoring='accuracy', verbose=1, n_jobs=-1)

grid_search.fit(X_train, y_train)

print(f"En iyi parametreler: {grid_search.best_params_}")
print(f"En iyi doğruluk skoru: {grid_search.best_score_}")

best_model = grid_search.best_estimator_

y_pred = best_model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"Test Doğruluğu: %{accuracy * 100:.2f}")

def classify_image(image_path, model):
    img = Image.open(image_path).convert('L')
    img_array = np.array(img).flatten()
    img_array = img_array.reshape(1, -1)

    prediction = model.predict(img_array)
    return prediction[0]

new_image_path = 'local/dentalModels/termal_images_split/termal16_3.png'
predicted_class = classify_image(new_image_path, best_model)
print(f"Tahmin edilen sınıf: {predicted_class}")

