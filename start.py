import tensorflow as tf
from tensorflow import keras
from keras.preprocessing.image import array_to_img, img_to_array, load_img
import numpy as np

# print(keras._version_)
model = keras.Sequential()
# first layer
model.add(keras.layers.Dense(256, activation='relu', input_shape=(32,32,3)))
model.add(keras.layers.Flatten())
#second layers
model.add(keras.layers.Dense(96, activation='relu'))
#softmax layers
model.add(keras.layers.Dense(46, activation='softmax'))
model.compile(optimizer=tf.train.AdamOptimizer(0.001), loss='categorical_crossentropy',metrics=[keras.metrics.categorical_accuracy])

batch_size = 16
test_datagen = keras.preprocessing.image.ImageDataGenerator(rescale=1./255)
train_generator = test_datagen.flow_from_directory(
        'uci/train',
        target_size=(32, 32),
        batch_size=20,
        class_mode='categorical')

print(train_generator.class_indices)

test_generator = test_datagen.flow_from_directory(
        'uci/test',
        target_size=(32, 32),
        batch_size=20,
        class_mode='categorical')

# validation_generator = test_datagen.flow_from_directory(
#         'vowels',
#         target_size=(36, 36),
#         batch_size=14,
#         class_mode='categorical')

import os.path
if os.path.exists('model.h5'):
    model.load_weights('model.h5')



model.fit_generator(
        train_generator,
        steps_per_epoch=3910 // batch_size,
        epochs=50,
        verbose=1,
        shuffle=True)
model.save_weights('model.h5')

score = model.evaluate_generator(
        test_generator,
        steps=3910, max_queue_size=10, workers=1, use_multiprocessing=False, verbose=1)

print('Test Loss :',score[0],'\nTest Accuracy :',score[1])

predict_generator = test_datagen.flow_from_directory(
        'uci/predict',
        target_size=(32, 32),
        batch_size=1,
        class_mode=None)
# print('\n\nType lol : ',type(predict_generator.class_indices))
# print('\n\ndict : ',predict_generator.class_indices)

predictions = model.predict_generator(predict_generator, steps=5)

# print('Prediction :',predictions)
# print('Shape : ',predictions.shape,'\nmax :',max(predictions),'type :',type(predictions))

def printPredictions(predict_dict,predictions):
        listofValues = np.argmax(predictions, axis=1).tolist()
        # print('\n\nlistofValues : ',listofValues,'\n')
        listofKeys = list(predict_dict.keys())
        predicted_label = []
        for value in listofValues:
                for key in listofKeys:
                        if(predict_dict[key]==value):
                                predicted_label.append(key)
                                break
        print(predicted_label)

printPredictions(train_generator.class_indices, predictions)
