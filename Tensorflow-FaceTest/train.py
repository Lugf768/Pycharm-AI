from keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D
from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
import matplotlib.pyplot as plt

batch_siz = 256
num_classes = 2
nb_epoch = 150
img_size = 48
data_path = './data'
model_path = './model'


class Model:
    def __init__(self):
        self.model = None

    def build_model(self):

        self.model = Sequential()

        self.model.add(Conv2D(32, (3, 3), strides=1, padding='same', input_shape=(img_size, img_size, 1)))
        self.model.add(Activation('relu'))
        self.model.add(Conv2D(32, (5, 5), padding='same'))
        self.model.add(Activation('relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))  # 池化，每个块只留下max

        self.model.add(Flatten())  # 扁平，折叠成一维的数组
        self.model.add(Dense(128))
        self.model.add(Activation('relu'))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(num_classes))
        self.model.add(Activation('softmax'))
        self.model.summary()  # 参数输出

    def train_model(self):
        opt = Adam(lr=0.0001)
        self.model.compile(optimizer=opt, loss="categorical_crossentropy", metrics=["accuracy"])
        # 自动扩充训练样本
        train_datagen = ImageDataGenerator(
            rescale=1. / 255,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True)
        # 归一化验证集
        val_datagen = ImageDataGenerator(
            rescale=1. / 255)
        # 以文件分类名划分label
        train_generator = train_datagen.flow_from_directory(
            data_path + '/train',
            target_size=(img_size, img_size),
            color_mode='grayscale',
            batch_size=batch_siz,
            class_mode='categorical')
        val_generator = val_datagen.flow_from_directory(
            data_path + '/val',
            target_size=(img_size, img_size),
            color_mode='grayscale',
            batch_size=batch_siz,
            class_mode='categorical')
        history_fit = self.model.fit_generator(
            train_generator,
            steps_per_epoch=10,
            nb_epoch=nb_epoch,
            validation_data=val_generator,
            validation_steps=100,
        )

        plt.plot(history_fit.history['loss'])
        plt.plot(history_fit.history['val_loss'])
        plt.title("model_loss")
        plt.ylabel("loss")
        plt.xlabel("nb_epoch")
        plt.legend(["loss", "val_loss"], loc="upper left")
        plt.show()

        plt.plot(history_fit.history['accuracy'])
        plt.plot(history_fit.history['val_accuracy'])
        plt.title("model_accuracy")
        plt.ylabel("accuracy")
        plt.xlabel("nb_epoch")
        plt.legend(["acc", "val_acc"], loc="upper left")
        plt.show()

        with open(model_path + '/model_fit_log', 'w') as f:
            f.write(str(history_fit.history))

    # 保存训练的模型文件
    def save_model(self):
        model_json = self.model.to_json()
        with open(model_path + "/model_json.json", "w") as json_file:
            json_file.write(model_json)
        self.model.save_weights(model_path + '/model_weight.h5')
        self.model.save(model_path + '/model.h5')


if __name__ == '__main__':
    model = Model()
    model.build_model()
    print('model built')
    model.train_model()
    print('model trained')
    model.save_model()
    print('model saved')
