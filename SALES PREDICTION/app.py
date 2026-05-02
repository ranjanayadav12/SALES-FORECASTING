# Import necessary libraries
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template
import pickle
from werkzeug.utils import secure_filename
import os

app = Flask(__name__)

# Load ML model
model = pickle.load(open('model.pkl', 'rb'))

# Global variable to store prediction history
prediction_history = []

# Folder to store uploaded files
UPLOAD_FOLDER = "uploads"
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Home route
@app.route('/')
def home():
    return render_template('index.html')

# Predict route (existing single-input form)
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # 1. Get inputs from form
        int_features = [float(x) for x in request.form.values()]

        # 2. Input validation
        if any(x < 0 for x in int_features):
            return render_template(
                'index.html',
                error="Please enter positive values only."
            )

        # 3. Convert to numpy array
        final_features = [np.array(int_features)]

        # 4. Predict using ML model
        prediction = model.predict(final_features)

        # 5. Round result
        output = round(prediction[0], 2)

        # 6. Save prediction history
        prediction_history.append(output)

        # 7. Send result + history to HTML
        return render_template(
            'index.html',
            prediction_text=f"Predicted Sales: ${output}",
            history=prediction_history
        )

    except:
        return render_template(
            'index.html',
            error="Invalid input. Please try again."
        )

# Results route for API requests
@app.route('/results', methods=['POST'])
def results():
    data = request.get_json(force=True)
    prediction = model.predict([np.array(list(data.values()))])
    output = prediction[0]
    return jsonify(output)

# Dashboard route for file upload
@app.route('/dashboard', methods=['GET', 'POST'])
def dashboard():
    if request.method == 'POST':
        # Check if file part exists
        if 'file' not in request.files:
            return render_template('dashboard.html', error="No file selected")

        file = request.files['file']
        if file.filename == '':
            return render_template('dashboard.html', error="No file selected")

        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        try:
            # Read uploaded CSV
            df = pd.read_csv(filepath)

            # Select numeric columns for prediction
            features = df.select_dtypes(include=['float64', 'int64']).values
            predictions = model.predict(features)

            # Add prediction column
            df['Predicted_Sales'] = predictions

            # If 'Cost' column exists, calculate profit/loss
            if 'Cost' in df.columns:
                df['Profit_Loss'] = df['Predicted_Sales'] - df['Cost']

            # Render table in dashboard
            return render_template(
                'dashboard.html',
                tables=[df.to_html(classes='table table-striped', index=False)],
                filename=filename
            )

        except Exception as e:
            return render_template('dashboard.html', error=f"Error processing file: {e}")

    # GET request
    return render_template('dashboard.html')

# Run Flask app
if __name__ == "__main__":
    app.run(debug=True)
