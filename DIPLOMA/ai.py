import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import cifar10
try:
    model = tf.keras.models.load_model('my_model.keras')
except Exception as e:
    model = None

print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
class_names = ['літак', 'автомобіль', 'птах', 'кіт', 'олень',
                   'собака', 'жаба', 'кінь', 'корабель', 'вантажівка']
(train_images, train_labels), (test_images, test_labels) = cifar10.load_data()

if not model:
    # Завантаження та підготовка даних

    train_images, test_images = train_images / 255.0, test_images / 255.0

    # Визначення класів


    # Побудова згорткової нейронної мережі (CNN)
    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(10))  # 10 класів

    # Компіляція моделі
    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])

    # Навчання моделі
    history = model.fit(train_images, train_labels, epochs=20,
                        validation_data=(test_images, test_labels))

    # Оцінка моделі
    test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)
    print(f'\nТочність на тестових даних: {test_acc}')

    model.save('my_model.keras')
# Використання моделі для передбачень

predictions = model.predict(test_images)
print(predictions)
# Відображення передбачення для першого зображення
plt.figure(figsize=(10, 10))
plt.imshow(test_images[100])
plt.title(f"Передбачений клас: {class_names[np.argmax(predictions[100])]}")
plt.show()
