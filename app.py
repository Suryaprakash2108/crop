from flask import Flask, render_template, request
import pandas as pd
import pickle
model = pickle.load(open('model.pkl', 'rb'))

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        features = [
            float(request.form['N']), float(request.form['P']), float(request.form['K']),
            float(request.form['temperature']), float(request.form['humidity']),
            float(request.form['ph']), float(request.form['rainfall'])
        ]
        prediction = model.predict([features])[0]
        return render_template('index.html', prediction=f'Recommended Crop: {prediction}')
    except Exception as e:
        return render_template('index.html', prediction=f'Error: {str(e)}')

if __name__ == '__main__':
    app.run(debug=True)
