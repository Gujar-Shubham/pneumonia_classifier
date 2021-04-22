#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = "Shubham Gujar"
__email__ = "gujar303shubham@gmail.com"
__website__ = "https://github.com/Gujar-Shubham/"

from flask_bootstrap import Bootstrap
from flask_wtf import FlaskForm
from flask_wtf.file import FileField, FileRequired, FileAllowed
from wtforms import SubmitField
import os
from werkzeug.utils import secure_filename
from flask import Flask, render_template
from tensorflow.keras.applications.vgg16 import decode_predictions, preprocess_input, VGG16
from tensorflow.keras.models import load_model
import cv2
import numpy as np


app = Flask(__name__)
app.config['SECRET_KEY'] = os.environ.get(
    'SECRET_KEY') or 'you-will-never-guess'
bootstrap = Bootstrap(app)
#model = VGG16(input_shape=(180, 180, 3), weights=None)
model = load_model('static/pvgg.h5')


class UploadForm(FlaskForm):
    upload = FileField('Select an image:', validators=[
        FileRequired(),
        FileAllowed(['jpg', 'png', 'jpeg', 'JPEG',
                     'PNG', 'JPG'], 'Images only!')
    ])
    submit = SubmitField('Classify')


def get_prediction(img_path):
    img = cv2.imread(img_path)
    img = cv2.resize(img, (180, 180))
    img = np.reshape(img, [1, 180, 180, 3])
    #classes = model.predict_classes(img)
    # print(classes)

    #img_resized = cv2.resize(img, (180, 180))
    #img_preprocessed = preprocess_input(img_resized)
    #img_reshaped = img_preprocessed.reshape((1, 180, 180, 3))
    preds = model.predict_classes(img)
    if(preds == [[1]]):
        pred = 'PNEUMONIA'
    else:
        pred = 'NORMAL'
    #print("Label name:", preds)
    return pred

#    prediction = model.predict(img_reshaped)
#    decoded = decode_predictions(prediction)

#    top_3 = [(cat.capitalize(), round(prob*100, 2)) for (code, cat, prob) in decoded[0]][:3]
#    return top_3


@app.route('/', methods=['GET', 'POST'])
def home():
    form = UploadForm()
    if form.validate_on_submit():
        f = form.upload.data
        filename = secure_filename(f.filename)
        file_url = os.path.join('static', filename
                                )
        f.save(file_url)
        form = None
        prediction = get_prediction(file_url)
    else:
        file_url = None
        prediction = None
    return render_template("index.html", form=form, file_url=file_url, prediction=prediction)


if __name__ == "__main__":
    app.run(debug=True)
