from flask import Flask, render_template, request
from utils import predict

app = Flask(__name__)

@app.route('/', methods=["GET"])
def hello_world():
    return render_template("index.html")

@app.route('/', methods=["POST"])
def diagnosis_prediction():
    imagefile = request.files["imagefile"]
    image_path = "./images/" + imagefile.filename
    imagefile.save(image_path)

    classification = predict(image_path)

    return render_template("index.html", prediction=classification)

if __name__ == '__main__':
    app.run(port=4444, debug=True)
