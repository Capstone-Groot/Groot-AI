import tensorflow as tf
import numpy as np
import tensorflow.keras as keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
from PIL import Image
import pathlib
import matplotlib.pyplot as plt
import os


# 학습, 검증, 데이터 부풀리기, 체크포인트와 모델 저장, 테스트 이미지 경로들 지정
train_dir = './aug_dataset'
train_dir = pathlib.Path(train_dir)
val_dir = './val_dataset'
val_dir = pathlib.Path(val_dir)
aug_dir = './dataset'
aug_dir = pathlib.Path(aug_dir)
aug_save_dir = './aug_dataset'
aug_save_dir = pathlib.Path(aug_save_dir)
checkpoint_dir = './checkpoints'
checkpoint_save_name = 'epoch-{epoch:02d}_acc-{val_accuracy:.2f}.h5'
# 저장된 모델 불러올 시 모델의 경로 입력
saved_model_path = ''
# 테스트 이미지 사용시 테스트 이미지의 경로 입력
test_img_path = ''


# checkpoint 폴더 없을시 생성
if not os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)

# 클래스 개수 읽기
train_CLASS_NAMES = np.array([item.name for item in train_dir.glob('*') if item.name != "LICENSE.txt"])
val_CLASS_NAMES = np.array([item.name for item in val_dir.glob('*') if item.name != "LICENSE.txt"])

# 훈련 파라미터 지정
IMG_HEIGHT = 224
IMG_WIDTH = 224
BATCH_SIZE = 16
EPOCH = 30


# 데이터 부풀리기
def data_aug():
    data_aug_generator = ImageDataGenerator(rescale=1. / 255,
                                            rotation_range=30,
                                            width_shift_range=0.15,
                                            height_shift_range=0.15,
                                            zoom_range=[0.9, 1.1],
                                            brightness_range=[0.8, 1.2],
                                            horizontal_flip=True,
                                            vertical_flip=False,
                                            fill_mode='constant')

    for category in aug_dir.glob('*'):
        if category.name != "LICENSE.txt":
            print("Processing..." + category.name)
            for file in category.glob('*.jpg'):
                img = load_img(file)
                x = img_to_array(img)
                x = x.reshape((1,) + x.shape)
                number_of_times = 0

                for batch in data_aug_generator.flow(x,
                                                     batch_size=1,
                                                     save_to_dir=aug_save_dir / category.name,
                                                     save_prefix=category.name,
                                                     save_format='jpg'):
                    number_of_times += 1
                    # number_of_times 는 한 이미지파일을 몇번 augmentation 할 것인가 지정
                    if number_of_times > 4:
                        break
    print("Augmentation Done!")


# 모델 만들기
def make_model():
    model = keras.Sequential()
    initializer = tf.keras.initializers.TruncatedNormal(mean=0., stddev=1.)


    model.add(keras.layers.Conv2D(filters=128, kernel_size=3, padding='SAME', kernel_initializer=initializer,
                                  input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Activation('relu'))
    model.add(keras.layers.Conv2D(filters=64, kernel_size=1, kernel_initializer=initializer, padding='SAME'))
    model.add(keras.layers.Conv2D(filters=32, kernel_size=1, kernel_initializer=initializer, padding='SAME'))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Activation('relu'))
    model.add(keras.layers.MaxPool2D(padding='SAME'))

    model.add(keras.layers.Conv2D(filters=256, kernel_size=3, kernel_initializer=initializer, padding='SAME'))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Activation('relu'))
    model.add(keras.layers.Conv2D(filters=128, kernel_size=1, kernel_initializer=initializer, padding='SAME'))
    model.add(keras.layers.Conv2D(filters=64, kernel_size=1, kernel_initializer=initializer, padding='SAME'))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Activation('relu'))
    model.add(keras.layers.MaxPool2D(padding='SAME'))

    model.add(keras.layers.Conv2D(filters=512, kernel_size=3, kernel_initializer=initializer, padding='SAME'))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Activation('relu'))
    model.add(keras.layers.Conv2D(filters=256, kernel_size=1, kernel_initializer=initializer, padding='SAME'))
    model.add(keras.layers.Conv2D(filters=128, kernel_size=1, kernel_initializer=initializer, padding='SAME'))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Activation('relu'))
    model.add(keras.layers.MaxPool2D(padding='SAME'))
    model.add(keras.layers.Conv2D(filters=1024, kernel_size=3, kernel_initializer=initializer, padding='SAME'))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Activation('relu'))
    model.add(keras.layers.GlobalAveragePooling2D(data_format='channels_last'))
    model.add(keras.layers.Dropout(0.4))
    model.add(keras.layers.Dense(len(train_CLASS_NAMES), activation='softmax'))

    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model


# 모델 학습하기
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
    # 콜백함수 생성
    model_callback = [keras.callbacks.EarlyStopping(monitor='val_accuracy',
                                                    mode='max',
                                                    patience=5,
                                                    verbose=1),
                      keras.callbacks.ModelCheckpoint(checkpoint_dir + '/' + checkpoint_save_name,
                                                      monitor='val_accuracy',
                                                      mode='max',
                                                      save_best_only=True,
                                                      save_freq='epoch',
                                                      verbose=1),
                      keras.callbacks.ReduceLROnPlateau(monitor='val_loss',
                                                        factor=0.75,
                                                        patience=1,
                                                        cooldown=3,
                                                        verbose=1,
                                                        min_lr=0.0001)]

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


# 테스트 이미지 예측
def predict(model):
    test_img = Image.open(test_img_path)
    test_img = test_img.resize((IMG_WIDTH, IMG_HEIGHT), Image.ANTIALIAS)
    test_img = img_to_array(test_img)
    test_img = test_img.reshape((1,) + test_img.shape)

    result = model.predict_classes(test_img)
    name = train_CLASS_NAMES
    result = int(result[0])
    return name[result]


# 데이터 부풀리기 실행
# data_aug()


# 모델 생성 및 학습
# model = make_model()
# fit_model(model)


# 저장된 모델을 불러와서 학습
# model = keras.models.load_model(saved_model_path)
# fit_model(model)


# 학습 된 모델로 테스트 이미지 예측
# model = keras.models.load_model(saved_model_path)
# print(predict(model))