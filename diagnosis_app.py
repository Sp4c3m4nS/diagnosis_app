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

    classification, scholar_results = predict(image_path)

    return render_template("index.html", prediction=classification, scholar=scholar_results)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=80, debug=True)
