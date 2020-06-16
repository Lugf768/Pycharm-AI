# Author LGF
# GreatTime 2020/5/20
# Description: simple introduction of the code
import cv2
import numpy as np
from keras.models import model_from_json

model_path = './model/'
img_size = 48
emotion_labels = ['natural', 'tired']
num_class = len(emotion_labels)

# 从json中加载模型
json_file = open(model_path + 'model_json.json')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)

# 加载模型权重
model.load_weights(model_path + 'model_weight.h5')
img = cv2.imread("./pic/t3.jpg")
# 使用opencv的人脸分类器
cascade = cv2.CascadeClassifier(model_path + 'haarcascade_frontalface_alt.xml')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# 检测人脸
faceLands = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=1, minSize=(120, 120))

if len(faceLands) > 0:
    for faceLand in faceLands:
        x, y, w, h = faceLand
        images = []
        result = np.array([0.0] * num_class)

        image = cv2.resize(gray[y:y + h, x:x + w], (img_size, img_size))
        image = image / 255.0
        image = image.reshape(1, img_size, img_size, 1)

        # 调用模型预测情绪，predict_proba返回的是一个 n 行 k 列的数组，列是标签（有排序）
        predict_lists = model.predict_proba(image, batch_size=32, verbose=1)
        # print(predict_lists)
        result += np.array([predict for predict_list in predict_lists for predict in predict_list])
        print(result)
        emotion = emotion_labels[int(np.argmax(result))]
        print("Emotion:", emotion)

cv2.imshow("Image", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
