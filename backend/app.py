from flask import Flask, render_template, request
import joblib
import numpy as np
import os

app = Flask(__name__, template_folder="../frontend")

# ---------- Load Model ----------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

model = joblib.load(os.path.join(BASE_DIR, "model", "xgboost_model.pkl"))
scaler = joblib.load(os.path.join(BASE_DIR, "model", "scaler.pkl"))
feature_order = joblib.load(os.path.join(BASE_DIR, "model", "feature_order.pkl"))

CLASS_MAP = {
    0: "Non-Habitable",
    1: "Potentially Habitable",
    2: "Highly Habitable"
}

# ---------- Habitability Score ----------
def compute_final_habitability_score(d):
    hsi = (
        0.25 * d["pl_rade"] +
        0.20 * d["pl_bmasse"] +
        0.20 * d["pl_dens"] +
        0.20 * d["pl_eqt"] +
        0.15 * d["pl_orbper"]
    )

    sci = (
        0.45 * d["st_spectral_score"] +
        0.25 * (1 - abs(d["st_teff"] - 0.5)) +
        0.20 * d["st_lum"] +
        0.10 * d["st_met"]
    )

    final = 0.8 * hsi + 0.2 * sci
    return round(min(max(final, 0), 1), 3)

# ---------- Routes ----------
@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    score = None

    if request.method == "POST":
        try:
            data = {f: float(request.form[f]) for f in feature_order}

            X = np.array([data[f] for f in feature_order]).reshape(1, -1)
            X_scaled = scaler.transform(X)

            pred = model.predict(X_scaled)[0]
            prob = model.predict_proba(X_scaled)[0].max()

            result = CLASS_MAP[int(pred)]
            score = compute_final_habitability_score(data)

        except Exception as e:
            result = f"Error: {e}"

    return render_template("index.html", result=result, score=score)

# ---------- Run ----------
if __name__ == "__main__":
    app.run(debug=True)
