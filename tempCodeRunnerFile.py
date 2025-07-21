from flask import Flask, render_template, request
import pandas as pd
import joblib

app = Flask(__name__)
model = joblib.load("model.pkl")
model_columns = joblib.load("model_columns.pkl")

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    confidence = None
    input_data = None

    if request.method == "POST":
        data = {
            "age": int(request.form["age"]),
            "education": request.form["education"],
            "occupation": request.form["occupation"],
            "gender": request.form["gender"],
            "hours-per-week": int(request.form["hours-per-week"])
        }
        input_df = pd.DataFrame([data])
        input_encoded = pd.get_dummies(input_df)
        input_encoded = input_encoded.reindex(columns=model_columns, fill_value=0)
        prediction_result = model.predict_proba(input_encoded)
        prediction = ">50K" if prediction_result[0][1] > 0.5 else "<=50K"
        confidence = round(prediction_result[0][1] * 100, 2)
        input_data = data

    return render_template("index.html", prediction=prediction, confidence=confidence, input_data=input_data)

if __name__ == "__main__":
    app.run(debug=True)
