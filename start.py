import tensorflow as tf
from tensorflow import keras
from keras.preprocessing.image import array_to_img, img_to_array, load_img


print(keras.__version__)
model = keras.Sequential()
# first layer
model.add(keras.layers.Dense(128, activation='relu', input_shape=(36,36,3)))
model.add(keras.layers.Flatten())
#second layers
model.add(keras.layers.Dense(64, activation='relu'))
#softmax layers
model.add(keras.layers.Dense(12, activation='softmax'))
model.compile(optimizer=tf.train.AdamOptimizer(0.001), loss='categorical_crossentropy',metrics=[keras.metrics.categorical_accuracy])

batch_size = 16
test_datagen = keras.preprocessing.image.ImageDataGenerator(rescale=1./255)
train_generator = test_datagen.flow_from_directory(
        'vowels',
        target_size=(36, 36),
        batch_size=14,
        class_mode='categorical')

import os.path
if os.path.exists('model.h5'):
    model.load_weights('model.h5')


model.fit_generator(
        train_generator,
        steps_per_epoch=460 // batch_size,
        epochs=50,
        verbose=1,
        shuffle=True)
model.save_weights('model.h5')
