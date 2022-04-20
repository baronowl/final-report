from keras.models import Sequential
from keras.layers import Conv1D, MaxPooling1D, LSTM
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.utils.vis_utils import plot_model
import os
import numpy as np

from model.monitor import TrainingMonitor, LossHistory,ModelCheckpoint


file_path = "result/cnn-lstm/" + "test_1" + "/"

if not os.path.exists(file_path):
    os.makedirs(file_path)
train_loss_path = file_path + "train_loss.txt"
validation_loss_path = file_path + "validation_loss.txt"
train_acc_path = file_path + "train_acc.txt"
validation_acc_path = file_path + "validation_acc.txt"

os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def data():
    X_train = np.load(
        "D:/#D/data/train_rri_ramp.npy")
    X_label = np.load("D:/#D/data/train_label.npy")
    Y_test = np.load("D:/#D/data/test_rri_ramp.npy")
    Y_label = np.load("D:/#D/data/test_label.npy")

    X_label = X_label.astype(dtype=np.int)
    Y_label = Y_label.astype(dtype=np.int)

    return X_train, X_label, Y_test, Y_label


def create_lstm_cnn_model(input_shape):
    model = Sequential()
    model.add(Conv1D(32, kernel_size=3, input_shape=input_shape, padding="valid", activation="relu"))
    model.add(MaxPooling1D(pool_size=2))


    model.add(Conv1D(64, kernel_size=3, padding="valid",activation="relu"))
    model.add(MaxPooling1D(pool_size=2))

    model.add(Dropout(0.25))

    model.add(Conv1D(128, kernel_size=3, padding="valid",activation="relu"))
    model.add(MaxPooling1D(pool_size=2))

    model.add(Dropout(0.25))

    model.add(LSTM(128, input_shape=input_shape, use_bias=True, dropout=0.25,
                   recurrent_dropout=0, return_sequences=True))

    model.add(LSTM(32, input_shape=input_shape, use_bias=True, dropout=0.25,
                   recurrent_dropout=0, return_sequences=True))

    model.add(Dense(32))

    model.add(Dense(1, activation="sigmoid"))

    # model.add(Conv1D(256, kernel_size=3, padding="valid",activation="relu"))
    # model.add(MaxPooling1D(pool_size=2))
    #
    # model.add(Dropout(0.25))


    model.add(Flatten())


    model.add(Dense(32, activation='relu'))
    model.add(Dense(1, activation="sigmoid"))

    model.compile(loss="binary_crossentropy", optimizer="adam", metrics=['accuracy'])

    model.summary()
    plot_model(model, to_file=file_path + '/cnn-lstm_model.png', show_shapes=True)

    return model


def trainning_model():
    print("getting data...")
    X_train, Y_train, X_test, Y_test = data()

    model = create_lstm_cnn_model(input_shape=(240, 3))
    fig_path = file_path
    model_path = file_path + "/model"

    if not os.path.exists(model_path):
        os.makedirs(model_path)
    model_path += "/model_{epoch:02d}-{val_accuracy:.6f}.hdf5"

    checkpoint = ModelCheckpoint(model_path, monitor='val_accuracy', verbose=1, save_best_only=True)
    callbacks = [
        TrainingMonitor(fig_path, model, train_loss_path, validation_loss_path, train_acc_path, validation_acc_path)
        , checkpoint]
    print("Training....")
    history = LossHistory()
    history.init()
    model.fit(X_train, Y_train, batch_size=12, epochs=15, callbacks=callbacks, validation_data=(X_test, Y_test))
    return model


if __name__ == '__main__':
    trainning_model()