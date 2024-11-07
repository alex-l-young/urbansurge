# CNN to identify all conduit diameters.

# Library imports.
import keras
import numpy as np

class CNNClassifier():
    def __init__(self, input_shape, num_classes, num_epochs, batch_size):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.num_epochs = num_epochs
        self.batch_size = batch_size

        # Instantiate the model.
        self.make_model()

    def make_model(self):
        input_layer = keras.layers.Input(self.input_shape)

        conv1 = keras.layers.Conv1D(filters=64, kernel_size=3, padding="same")(input_layer)
        # conv1 = keras.layers.BatchNormalization()(conv1)
        conv1 = keras.layers.ReLU()(conv1)

        conv2 = keras.layers.Conv1D(filters=64, kernel_size=3, padding="same")(conv1)
        # conv2 = keras.layers.BatchNormalization()(conv2)
        conv2 = keras.layers.ReLU()(conv2)

        conv3 = keras.layers.Conv1D(filters=64, kernel_size=3, padding="same")(conv2)
        # conv3 = keras.layers.BatchNormalization()(conv3)
        conv3 = keras.layers.ReLU()(conv3)

        gap = keras.layers.GlobalAveragePooling1D()(conv3)

        output_layer = keras.layers.Dense(self.num_classes, activation="softmax")(gap)

        self.model = keras.models.Model(inputs=input_layer, outputs=output_layer)

    def fit_model(self, X_train, y_train):
        callbacks = [
            keras.callbacks.ModelCheckpoint(
                "model_saves/best_model_cnn.keras", save_best_only=True, monitor="val_loss"
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss", factor=0.5, patience=20, min_lr=0.0001
            ),
            keras.callbacks.EarlyStopping(monitor="val_loss", patience=50, verbose=1),
        ]
        self.model.compile(
            optimizer="adam",
            loss="sparse_categorical_crossentropy",
            metrics=["sparse_categorical_accuracy"],
        )

        history = self.model.fit(
            X_train,
            y_train,
            batch_size=self.batch_size,
            epochs=self.num_epochs,
            callbacks=callbacks,
            validation_split=0.15,
            verbose=1,
        )

        return history

    def predict(self, X_test, y_test):
        model = keras.models.load_model("model_saves/best_model_cnn.keras")

        test_loss, test_acc = model.evaluate(X_test, y_test)

        y_pred = model.predict(X_test)

        return y_pred, test_loss, test_acc


if __name__ == '__main__':
    def readucr(filename):
        data = np.loadtxt(filename, delimiter="\t")
        y = data[:, 0]
        x = data[:, 1:]
        return x, y.astype(int)


    root_url = "https://raw.githubusercontent.com/hfawaz/cd-diagram/master/FordA/"

    x_train, y_train = readucr(root_url + "FordA_TRAIN.tsv")
    x_test, y_test = readucr(root_url + "FordA_TEST.tsv")

    classes = np.unique(np.concatenate((y_train, y_test), axis=0))

    x_train = x_train.reshape((x_train.shape[0], x_train.shape[1], 1))
    x_test = x_test.reshape((x_test.shape[0], x_test.shape[1], 1))

    idx = np.random.permutation(len(x_train))
    x_train = x_train[idx]
    y_train = y_train[idx]
    y_train[y_train == -1] = 0
    y_test[y_test == -1] = 0

    classifier = CNNClassifier(x_train.shape[1:], len(np.unique(y_train)), 200, 32)
    classifier.fit_model(x_train, y_train)

    classifier.predict(x_test, y_test)