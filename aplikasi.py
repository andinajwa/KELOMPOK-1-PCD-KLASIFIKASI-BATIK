from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

app = Flask(__name__)

dic = {0: 'Cendrawasih', 1: 'Kawung', 2: 'Megamendung', 3: 'Parang'}

model = load_model('klasifikasi_batik_model_exception.h5')

def predict_label(img_path):
    i = image.load_img(img_path, target_size=(128, 128))
    i = image.img_to_array(i) / 255.0
    i = np.expand_dims(i, axis=0)
    p = model.predict(i)
    return dic[np.argmax(p)]

# routes
@app.route("/", methods=['GET', 'POST'])
def main():
    return render_template("classification.html")

@app.route("/submit", methods = ['GET', 'POST'])
def get_output():
    if request.method == 'POST':
        img = request.files['my_image']
        img_path = "static/" + img.filename    
        img.save(img_path)
        p = predict_label(img_path)
        return render_template("classification.html", prediction = p, img_path = img_path)

if __name__ == '__main__':
    app.run(debug=True)
