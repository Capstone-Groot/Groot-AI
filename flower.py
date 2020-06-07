import tensorflow as tf
import numpy as np
import tensorflow.keras as keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from PIL import Image
import pathlib
import matplotlib.pyplot as plt

#학습, 검증, 데이터 부풀리기 경로들 지정
# train_dir = 'C:/Users/js022/Desktop/flower/flower'
train_dir = 'C:/Users/js022/Desktop/flower_photos/flower_photos'
train_dir = pathlib.Path(train_dir)
val_dir = 'C:/Users/js022/Desktop/flower/flower_val'
val_dir = pathlib.Path(val_dir)
aug_dir = 'C:/Users/js022/Desktop/flower_photos/flower_photos'
aug_dir = pathlib.Path(aug_dir)
aug_save_dir = 'C:/Users/js022/Desktop/flower/flower'
aug_save_dir = pathlib.Path(aug_save_dir)

#클래스 개수 읽기
train_CLASS_NAMES = np.array([item.name for item in train_dir.glob('*') if item.name != "LICENSE.txt"])
val_CLASS_NAMES = np.array([item.name for item in val_dir.glob('*') if item.name != "LICENSE.txt"])

#파라미터 지정
IMG_HEIGHT = 224
IMG_WIDTH = 224
BATCH_SIZE = 16
EPOCH = 15

#데이터 부풀리기
def data_aug():
    number_of_times = 3

    data_aug_generator = ImageDataGenerator(rescale=1. / 255,
                                            rotation_range=30,
                                            width_shift_range=0.15,
                                            height_shift_range=0.15,
                                            zoom_range=[0.8, 1.1],
                                            horizontal_flip=True,
                                            vertical_flip=False,
                                            fill_mode='constant')

    for category in aug_dir.glob('*'):
        if category.name != "LICENSE.txt":
            for file in category.glob('*.jpg'):
                img = load_img(file)
                print("Processing..." + str(file))
                x = img_to_array(img)
                x = x.reshape((1,) + x.shape)
                i = 0
                for batch in data_aug_generator.flow(x,
                                            batch_size=1,
                                            save_to_dir=aug_save_dir / category.name,
                                            save_prefix=category.name,
                                            save_format='jpg'):
                    i += 1
                    if i > 3:
                        break

#모델 만들기
def make_model():
    model = keras.Sequential()  # Sequential API 생성

    model.add(keras.layers.Conv2D(filters=128, kernel_size=3, activation='relu', padding='SAME',
                                  input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)))  # 처음에만 input_shape 정의
    model.add(keras.layers.MaxPool2D(padding='SAME'))  # 디폴트 = 사이즈(2X2), strides:2

    model.add(keras.layers.Conv2D(filters=256, kernel_size=3, activation='relu', padding='SAME'))
    model.add(keras.layers.MaxPool2D(padding='SAME'))

    model.add(keras.layers.Conv2D(filters=512, kernel_size=3, activation='relu', padding='SAME'))
    model.add(keras.layers.MaxPool2D(padding='SAME'))

    model.add(keras.layers.Conv2D(filters=1024, kernel_size=3, activation='relu', padding='SAME'))
    model.add(keras.layers.MaxPool2D(padding='SAME'))

    model.add(keras.layers.Flatten())  # 벡터 평탄화

    model.add(keras.layers.Dense(256, activation='relu'))

    model.add(keras.layers.Dropout(0.4))  # 트레이닝 과정에서만 Dropout

    model.add(keras.layers.Dense(len(train_CLASS_NAMES), activation='softmax'))  # 최종 output 레이블 개수 : 10

    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model

#모델 학습하기
def fit_model(model):
    train_image_count = len(list(train_dir.glob('*/*.jpg')))
    val_image_count = len(list(val_dir.glob('*/*.jpg')))

    train_image_generator = ImageDataGenerator(rescale=1. / 255)
    validation_image_generator = ImageDataGenerator(rescale=1. / 255)

    train_data_gen = train_image_generator.flow_from_directory(directory=str(train_dir),
                                                               batch_size=BATCH_SIZE,
                                                               shuffle=True,
                                                               target_size=(IMG_HEIGHT, IMG_WIDTH),
                                                               classes=list(train_CLASS_NAMES))
    val_data_gen = validation_image_generator.flow_from_directory(batch_size=BATCH_SIZE,
                                                                  directory=str(val_dir),
                                                                  target_size=(IMG_HEIGHT, IMG_WIDTH),
                                                                  classes=list(val_CLASS_NAMES))

    model_callback = [keras.callbacks.EarlyStopping(monitor='val_accuracy',
                                                    mode='max',
                                                    patience=2,
                                                    verbose=1),
                      keras.callbacks.ModelCheckpoint('C:/Users/js022/PycharmProjects/keras/checkpoints/ckpt_acc.h5',
                                                      monitor='val_accuracy',
                                                      mode='max',
                                                      save_best_only=True,
                                                      save_freq='epoch',
                                                      verbose=1)]

    history = model.fit(
        train_data_gen,
        steps_per_epoch=train_image_count // BATCH_SIZE,
        epochs=EPOCH,
        validation_data=val_data_gen,
        validation_steps=val_image_count // BATCH_SIZE,
        verbose=2,
        callbacks=model_callback)
    model.summary()
    
    # 학습 정확도와 검증 정확도를 플롯팅.
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.show()

    # 학습 손실 값과 검증 손실 값을 플롯팅.
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.show()

#테스트 이미지 예측
def predict():
    test_img = Image.open('C:/Users/js022/Desktop/sunflower_test2.jpg')
    test_img = test_img.resize((224,224), Image.ANTIALIAS)
    test_img = img_to_array(test_img)
    test_img = test_img.reshape((1,) + test_img.shape)

    model = make_model()
    model.load_weights('C:/Users/js022/PycharmProjects/keras/checkpoints/ckpt_loss2.h5')
    result = model.predict_classes(test_img)
    name = ['daisy', 'dandelion', 'dandelion', 'rose', 'sunflower', 'tulip']
    result = int(result[0])
    return name[result]
