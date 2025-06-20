import os
import cv2
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
import joblib

# Папканын дареги
dataset_path = 'dataset'

# Урууларды окуу (каталогдор)
tribes = sorted(os.listdir(dataset_path))

# Дайындоо үчүн бош тизмелер
X = []
y = []

# Ар бир уруунун сүрөттөрүн окуу
for label, tribe in enumerate(tribes):
    tribe_path = os.path.join(dataset_path, tribe)
    if not os.path.isdir(tribe_path):
        continue

    for image_name in os.listdir(tribe_path):
        image_path = os.path.join(tribe_path, image_name)
        try:
            img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img, (50, 50))  # Бирдей өлчөмдө
            X.append(img.flatten())
            y.append(label)
        except Exception as e:
            print(f"Сүрөт иштетүүдө ката чыкты: {image_path}, ката: {e}")

# NumPy массивдер
X = np.array(X)
y = np.array(y)

# Модель түзүү жана машыгуу
model = KNeighborsClassifier(n_neighbors=3)
model.fit(X, y)

# Моделди сактоо
joblib.dump(model, 'muras.pkl')

print("✅ Модель ийгиликтүү машыгып жана muras.pkl файлына сакталды.")
