import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Загрузка обученной модели
model = load_model("model.h5")

# Путь к изображению для тестирования
test_image_path = "test4.jpg"  # Замените на путь к вашему тестовому изображению

# Загрузка и предобработка изображения
def preprocess_image(image_path):
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Переключение цветовых каналов на RGB
    img = cv2.resize(img, (224, 224))  # Используем размер, подходящий для модели
    img = img / 255.0  # Нормализация значений пикселей к диапазону [0, 1]
    img = np.expand_dims(img, axis=0)  # Добавление дополнительной размерности для батча
    return img

# Загрузка и предобработка тестового изображения
processed_image = preprocess_image(test_image_path)

# Получение предсказаний от модели
predictions = model.predict(processed_image)

# Индекс с наивысшей уверенностью
predicted_class_index = np.argmax(predictions)

# Загрузка словаря с соответствием классов и индексов
class_indices = {0: 'circle', 1: 'rhombus', 2: 'square', 3: 'star', 4: 'triangle'}

# Получение названия класса
predicted_class_name = class_indices[predicted_class_index]

# Вывод результата
print(f"Predicted class: {predicted_class_name}")
