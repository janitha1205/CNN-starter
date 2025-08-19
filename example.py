
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# to get available data test for studies perpose
from tensorflow.keras.datasets import mnist

# to convert 0 t0 9 in digit to binary class as 10 dim binary numbers to deferentiate the perpose of common nueral network training perposes
from tensorflow.keras.utils import to_categorical

# build a model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPool2D, Flatten

# fitting data data
from tensorflow.keras.callbacks import EarlyStopping

(x_tr, y_tr), (x_tes, y_tes) = mnist.load_data()

print(x_tr.shape)

plt.imshow(x_tr[0])
plt.show()

y_cat_tr = to_categorical(y_tr)
y_cat_tes = to_categorical(y_tes)

print(y_tr.shape)
print(y_cat_tr.shape)

# input data should b e set t0 in a rage of 1 to 0

x_tr = x_tr / 255
x_tes = x_tes / 255
# nothing change in appearance as if change the range from (0 to 255) to (0 to 1)
plt.imshow(x_tr[0])
plt.show()

# reshape the image structure to col by row by numchanels(to 1/ for RGB its 3)
# usually CNN needs aditional like chanels as for default
x_tr = x_tr.reshape(x_tr.shape[0], x_tr.shape[1], x_tr.shape[2], 1)
print(x_tr.shape)

x_tes = x_tes.reshape(x_tes.shape[0], x_tes.shape[1], x_tes.shape[2], 1)
print(x_tes.shape)

# model building

model = Sequential()

model.add(
    Conv2D(
        filters=32,
        kernel_size=(4, 4),
        input_shape=(x_tr.shape[1], x_tr.shape[2], x_tr.shape[3]),
        activation="relu",
    )
)
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(128, activation="relu"))
model.add(Dense(10, activation="softmax"))

model.summary()

early_stop = EarlyStopping(monitor="val_loss", patience=2)

model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

# train the model

model.fit(
    x_tr,
    y_cat_tr,
    epochs=10,
    validation_data=(x_tes, y_cat_tes),
    callbacks=early_stop,
    verbose=True,
)


# model.metrics_names["loss", "accuracy"]
losses = pd.DataFrame(model.history.history)
print(losses)
losses[["loss", "val_loss"]].plot()
losses[["accuracy", "val_accuracy"]].plot()
model.evaluate(x_tes, y_cat_tes)
y_p = model.predict(x_tes[0].reshape(1, 28, 28, 1))

y_class = np.argmax(y_p, axis=1)
print(y_class)
