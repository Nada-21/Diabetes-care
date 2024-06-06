from flask import Flask, render_template, request
import numpy as np
import pickle


app = Flask(__name__,template_folder="template")
model = pickle.load(open("diabetes_model.sav", "rb"))



@app.route('/')
def hello_name():
   return render_template('index.html')

@app.route("/predict", methods=["POST"])
def predict():

    int_features = [float(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)

    output = prediction[0]

    countries = ["No Diabetes", "Diabetes"]

    return render_template(
        "index.html", prediction_text="{}".format(countries[output])
    )

if __name__ == '__main__':
   app.run(debug=True,port=9000)
   