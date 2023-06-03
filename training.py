import os, cv2, time
import tensorflow as tf
from tensorflow.keras import layers
import matplotlib.pyplot as plt
from tensorflow.keras import regularizers

from tensorflow.python.client import device_lib

print(device_lib.list_local_devices())

# Train/test set ratio
test_set_size = 0.2
# Using diff batches
# Propagation & Batch size
# See https://stats.stackexchange.com/a/153535
batch_size = 40
image_size = (230,230)  # width  # height

order = None  # ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11"]

train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    "data2",
    validation_split=test_set_size,
    subset="training",
    seed=3489,
    color_mode="rgb",
    image_size=image_size,
    batch_size=batch_size,
    class_names=order,
)

test_ds = tf.keras.preprocessing.image_dataset_from_directory(
    "data2",
    validation_split=test_set_size,
    subset="validation",
    seed=3489,
    color_mode="rgb",
    image_size=image_size,
    batch_size=batch_size,
    class_names=order,
)
class_names = train_ds.class_names
print(class_names)
# class_names = ["Dairy", "Dessert", "Egg", "Fast Food", "Meat", "Noodles","Rice", "Seafood", "Soup", "Fresh Produce", "Bread"]
"""

plt.figure(figsize=(10, 10))
for images, labels in train_ds.take(1):
    for i in range(9):
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(images[i].numpy().astype("uint8"))
        plt.title(class_names[labels[i]])
        plt.axis("off")

"""

regularise = {
"kernel_regularizer": regularizers.L1L2(l1=1e-5, l2=1e-3),
"bias_regularizer": regularizers.L2(1e-4),
"activity_regularizer": regularizers.L2(1e-5)}

# Sequential Model
no_neurons_per_layer = 300
dropout_rate = 0.20
learning_rate = 0.004
filters = 16
model = tf.keras.Sequential(
    [
        tf.keras.Input(shape=(*image_size, 3)),  # 1 for B/W, 3, for Color/RGB
        layers.RandomFlip("horizontal"),
        # # layers.RandomRotation(0.2),

        layers.Rescaling(1.0 / 127.5, offset=-1),
        layers.Conv2D(filters, 3, padding="same", activation="swish", **regularise),  # Convolution
        layers.MaxPooling2D(pool_size=[2, 2], strides=2),
        layers.Conv2D(2 * filters, 3, padding="same", activation="swish", **regularise),  # Convolution
        layers.MaxPooling2D(pool_size=[2, 2], strides=2),
        # layers.Conv2D(4 * filters, 3, padding="same", activation="swish", **regularise),  # Convolution
        layers.Conv2D(4 * filters, 3, padding="same", activation="swish", **regularise),  # Convolution
        layers.MaxPooling2D(pool_size=[2, 2], strides=2),
        layers.Flatten(),
        # https://stackoverflow.com/a/56797269/12204281        
        layers.Dropout(rate=dropout_rate),
        layers.Dense(no_neurons_per_layer, activation="swish", **regularise),
        layers.Dropout(rate=dropout_rate),
        # layers.Dense(no_neurons_per_layer, activation="swish", **regularise),
        # layers.Dropout(rate=dropout_rate),
        # layers.Dense(no_neurons_per_layer, activation="swish", **regularise), # softmax on the last layer
        
        layers.Dense(9, activation="softmax", **regularise),
    ],
    name="FoodClassifier",
)



"""
Questions:
- Diff between the optimisers?
- Losses and types
-
"""
model.compile(
    optimizer=tf.keras.optimizers.Adamax(
        learning_rate=learning_rate
    ),  # gradient descent learning rate
    loss=tf.losses.SparseCategoricalCrossentropy(
        from_logits=False
    ),  # What is y_pred???? - predicted value
    metrics=["accuracy"],  # Any other metrics?
)
model.summary()

callbacks =  [
    tf.keras.callbacks.EarlyStopping(monitor='val_binary_crossentropy', patience=200)]


# while True:
#     plt.waitforbuttonpress()
model.fit(train_ds, validation_data=test_ds, epochs=20, callbacks=callbacks)
model.save(f"model10.h5")

# model.predict()