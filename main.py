import cv2
import matplotlib.pyplot as plt


def detect_face(img):
    # Загрузка каскада Хаара для поиска лиц
    classifier = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    # выполнение распознавания лиц
    bboxes = classifier.detectMultiScale(img, scaleFactor=2, minNeighbors=3, minSize=(50, 50))
    return bboxes


def detect_eyes(img, bboxes):
    # Загрузка каскада Хаара для поиска глаз
    classifier_eye = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")
    eyes = []
    for box in bboxes:
        x, y, width, height = box
        face = img[y:y + height, x:x + width]
        eyes_in_face = classifier_eye.detectMultiScale(face)
        for (x_eye, y_eye, w_eye, h_eye) in eyes_in_face:
            eyes.append((x + x_eye, y + y_eye, w_eye, h_eye))
    return eyes


def draw_ovals(img, bboxes, eyes):
    for box in bboxes:
        x, y, width, height = box
        # рисуем овал вокруг лица
        cv2.ellipse(img, (int(x + width / 2), int(y + height / 2)), (int(width / 2), int(height / 1)), 0, 0, 360,
                    (0, 255, 0), 2)
    for eye in eyes:
        x, y, width, height = eye
        # рисуем круг вокруг глаза
        cv2.circle(img, (int(x + width / 2), int(y + height / 2)), int(width / 2), (0, 0, 255), 2)
    return img


# загружаем изображение
img = cv2.imread('./jason.jpg')
# копируем исходное изображение
img_copy = img.copy()
# выполнение распознавания лиц
bboxes = detect_face(img)
# выполнение распознавания глаз
eyes = detect_eyes(img, bboxes)
# рисование овалов вокруг лиц и глаз
result_img = draw_ovals(img_copy, bboxes, eyes)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 8))
ax1.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
ax1.xaxis.set_ticks([])
ax1.yaxis.set_ticks([])
ax1.set_title('Исходное изображение')

ax2.imshow(cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB))
ax2.xaxis.set_ticks([])
ax2.yaxis.set_ticks([])
ax2.set_title('Распознанные лица и глаза')

plt.show()
