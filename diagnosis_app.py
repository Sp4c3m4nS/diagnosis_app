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

    scan = request.form["scan"] 

    classification, scholar1, scholar2, scholar3 = predict(image_path, scan)

    return render_template("index.html", prediction=classification, scholar1=scholar1, scholar2=scholar2, scholar3=scholar3)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=80, debug=True)
