import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import VGG16
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Предполагаем, что у вас есть директория с изображениями для каждого класса в формате "Circle", "Rhombus", и т.д.
train_dir = "dataset"
batch_size = 32
img_size = (224, 224)  # Используем размер, подходящий для VGG16

# Создаем генераторы изображений
train_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical',
    subset='training'
)

validation_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation'
)

# Загрузка предварительно обученной модели VGG16 без верхних слоев
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(img_size[0], img_size[1], 3))

# Заморозка весов базовой модели
base_model.trainable = False

# Создание новой модели с базовой моделью VGG16 и дополнительными слоями
model = keras.Sequential([
    base_model,
    layers.Flatten(),
    layers.Dense(256, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(5, activation='softmax')  # Учтите, что у вас 5 классов
])

# Компиляция модели
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Обучение модели
epochs = 10
model.fit(
    train_generator,
    epochs=epochs,
    validation_data=validation_generator
)

# Сохранение модели в файл
model.save("model.h5")
