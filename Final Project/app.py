# from flask import Flask, render_template, request, url_for
# import os
# import cv2
# import numpy as np
# from sklearn.svm import SVC
# from sklearn.model_selection import train_test_split

# app = Flask(__name__)
# app.config['UPLOAD_FOLDER'] = 'static/uploads'

# # Dataset жана уруунун тизмеси
# DATASET_PATH = 'dataset'
# tribe_names = sorted(os.listdir(DATASET_PATH))

# X, y = [], []
# for label, tribe in enumerate(tribe_names):
#     path = os.path.join(DATASET_PATH, tribe)
#     for img in os.listdir(path):
#         img_path = os.path.join(path, img)
#         image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
#         if image is not None:
#             image = cv2.resize(image, (50, 50)).flatten()
#             X.append(image)
#             y.append(label)

# X = np.array(X)
# y = np.array(y)

# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
# model = SVC(kernel='linear')
# model.fit(X_train, y_train)

# # Уруулар тууралуу маалыматтар
# tribe_info = {
#     "munduz": "Мундуз уруусу — кыргыздардын белгилүү урууларынын бири. Алардын тарыхы жана каада-салты терең.",
#     "sarybagysh": "Сарыбагыш — көрүнүктүү уруулардын бири, көптөгөн тарыхый инсандар чыккан.",
#     "buuju": "Буужу уруусу өзүнүн тарыхы жана эли менен айырмаланат.",
#     # дагы урууларды кош...
# }

# def predict_tribe(image_path):
#     img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
#     img = cv2.resize(img, (50, 50)).flatten().reshape(1, -1)
#     pred = model.predict(img)
#     return tribe_names[pred[0]]

# @app.route('/')
# def index():
#     return render_template('index.html')

# @app.route('/upload', methods=['POST'])
# def upload():
#     file = request.files['file']
#     if file:
#         filename = file.filename
#         path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
#         file.save(path)
#         result = predict_tribe(path)
#         return render_template('result.html', result=result, filename=filename)
#     return "Файл жүктөлгөн жок"

# @app.route('/tribe/<tribe_name>')
# def tribe_info_page(tribe_name):
#     description = tribe_info.get(tribe_name, "Маалымат табылган жок.")
#     return render_template('tribe_info.html', tribe_name=tribe_name, description=description)

# if __name__ == '__main__':
#     app.run(debug=True)
