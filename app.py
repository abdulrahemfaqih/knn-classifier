from flask import Flask, request, render_template, redirect, url_for
import pickle

app = Flask(__name__)
application = app

# Load the trained model and scaler
with open("model/knn_pickle", "rb") as r:
    knnp = pickle.load(r)

with open("model/scaler_pickle", "rb") as s:
    scaler = pickle.load(s)

LABEL = ["Dropout", "Graduate"]

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    course = int(request.form['course'])
    bayar = int(request.form['bayar'])
    gender = int(request.form['gender'])
    bea = int(request.form['bea'])
    ip1 = float(request.form['ip1'])
    nilai1 = float(request.form['nilai1'])
    ip2 = float(request.form['ip2'])
    nilai2 = float(request.form['nilai2'])

    newdata = [[course, bayar, gender, bea, ip1, nilai1, ip2, nilai2]]
    newdata_scaled = scaler.transform(newdata)
    result = knnp.predict(newdata_scaled)
    result = LABEL[result[0]]

    return render_template('result.html',
                           course=course, bayar=bayar, gender=gender, bea=bea,
                           ip1=ip1, nilai1=nilai1, ip2=ip2, nilai2=nilai2, result=result)

if __name__ == "__main__":
    app.run(debug=True)
