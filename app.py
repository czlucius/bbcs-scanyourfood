import tensorflow as tf
import cv2, os
import numpy as np
from tensorflow.python.client import device_lib
from flask import (
    Flask,
    render_template,
    request,
    redirect,
    url_for,
    send_from_directory,
)

print(device_lib.list_local_devices())

learning_rate = 0.004
model = tf.keras.models.load_model("model7.h5", compile=False)
model.compile(
    optimizer=tf.keras.optimizers.Adamax(
        learning_rate=learning_rate
    ),  # gradient descent learning rate
    loss=tf.losses.SparseCategoricalCrossentropy(
        from_logits=True
    ),  # What is y_pred???? - predicted value
    metrics=["accuracy"],  # Any other metrics?
)

from flask import Flask

app = Flask(
    __name__,
    template_folder="./template",
    static_folder="./static",
)

app.config["IMAGE_UPLOADS"] = r"H:\Lucius\code\BBCS2023\upload"


@app.route("/hello")
def hello_world():
    return "<p>Hello, World!</p>"


@app.route("/", methods=["POST"])
def upload_file():
    if request.method == "POST":
        f = request.files["file"]
        file_ext = os.path.splitext(f.filename)[1]
        f.save(os.path.join(app.config["IMAGE_UPLOADS"], "upload_img"))
        return redirect(url_for("index"))

class_names=['dairy', 'egg', 'fast_food', 'meat', 'noodles', 'rice', 'seafood', 'soup', 'wheat']
@app.route("/")
def index():
    img = cv2.imread("upload/upload_img")
    img = cv2.resize(img, dsize=(230, 230))

    img = np.expand_dims(img, axis=0)
    out = model.predict(img)
    preds = out[0]
    likely = []
    i=0
    for pred in preds:
        print(pred, class_names[i], float(pred), float(pred)>=0.05)
        if float(pred) >= 0.05: # Not a rare event
            likely.append(f"{class_names[i]} {round(float(pred), 4)*100}%")
        i+=1



    return render_template("template.html", file="upload_img", prediction=" ".join(likely))


@app.route("/upload/<name>")
def download_file(name):
    return send_from_directory(app.config["IMAGE_UPLOADS"], name)
