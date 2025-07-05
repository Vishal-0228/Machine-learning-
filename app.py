from flask import Flask, render_template, request
import joblib
import numpy as np
import os

# Initialize Flask app
app = Flask(__name__, template_folder="templates")

# Load trained model
try:
    model_path = os.path.join(os.path.dirname(__file__), "..", "model", "covid.pkl")
    model = joblib.load(model_path)
except Exception as e:
    print(f"Error loading model: {e}")
    model = None  # Prevents crashing if model not found

# Feature order
feature_names = [
    "Breathing Problem", "Fever", "Dry Cough", "Sore throat", "Running Nose",
    "Asthma", "Headache", "Heart Disease", "Diabetes",
    "Hyper Tension", "Fatigue", "Abroad travel",
    "Contact with COVID Patient", "Attended Large Gathering",
    "Visited Public Exposed Places", "Family working in Public Exposed Places",
    "Wearing Masks", "Sanitization from Market"
]

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if model is None:
            return "Error: Model not loaded properly."

        form_data = []

        for feature in feature_names:
            value = request.form.get(feature, "").strip().lower()
            print(f"{feature}: {value}")  # Debugging line to check input values
            
            if value not in ["yes", "no"]:  
                return f"Error: Invalid input for '{feature}'. Enter 'Yes' or 'No' only."

            form_data.append(1 if value == "yes" else 0)

        input_array = np.array([form_data])

        prediction = model.predict(input_array)
        #result = "COVID Positive" if int(prediction[0]) == 1 else "COVID Negative"
        prediction_result = int(prediction[0])

        if prediction_result == 1:
            result = "COVID Positive"
            precautions = [
                "Self-isolate immediately",
                "Consult a healthcare provider",
                "Monitor your symptoms",
                "Stay hydrated and rest",
                "Wear a mask to prevent spread",
                "Avoid contact with others, especially the elderly"
            ]
        else:
            result = "COVID Negative"
            precautions = [
                "Continue following safety guidelines",
                "Wear masks in public places",
                "Maintain social distancing",
                "Avoid large gatherings",
                "Sanitize your hands regularly",
                "Stay informed and vaccinated"
            ]



        return render_template('result.html', prediction=result, precautions=precautions)

    except Exception as e:
        print(f"Error in prediction: {e}")
        return f"Error: {str(e)}"

if __name__ == "__main__":
    app.run(debug=True)
