from flask import Flask, render_template, request
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import os

app = Flask(__name__)
model = load_model("model.h5")  # Ensure model expects 100x100 grayscale input

@app.route("/", methods=["GET", "POST"])
def index():
    prediction_text = ""
    confidence = ""
    color = ""

    if request.method == "POST":
        file = request.files["file"]
        if file:
            file_path = os.path.join("static", file.filename)
            file.save(file_path)

            # Preprocess the image for prediction
            img = image.load_img(file_path, target_size=(100, 100), color_mode="grayscale")
            img_array = image.img_to_array(img)
            img_array = img_array / 255.0
            img_array = np.expand_dims(img_array, axis=0)

            # Make prediction
            prediction = model.predict(img_array)[0][0]

            # Interpret results
            if prediction > 0.5:
                prediction_text = "🚨 Positive"
                confidence = f"Confidence: {round(prediction * 100, 2)}%"
                color = "red"
            else:
                prediction_text = "✅ Negative"
                confidence = f"Confidence: {round((1 - prediction) * 100, 2)}%"
                color = "green"

            return render_template(
                "index.html",
                prediction_text=prediction_text,
                confidence=confidence,
                color=color,
                image_path=file_path
            )

    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)
