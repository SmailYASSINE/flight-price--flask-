import pickle

import numpy as np
from flask import Flask, request, render_template

app = Flask(__name__)
model = pickle.load(open('C:\\Users\\um6p\\Desktop\\M1\\M2\\kelloubi\\TP1\\model.pkl', 'rb'))

# list of one-hot encoded airline options
airline_options = {
    'Air India': [1, 0, 0, 0, 0],
    'GO FIRST': [0, 1, 0, 0, 0],
    'Indigo': [0, 0, 1, 0, 0],
    'SpiceJet': [0, 0, 0, 1, 0],
    'Vistara': [0, 0, 0, 0, 1]
}

# list of one-hot encoded flight options
flight_options = {
    'UK-720': [1, 0, 0, 0, 0],
    'UK-822': [0, 1, 0, 0, 0],
    'UK-826': [0, 0, 1, 0, 0],
    'UK-828': [0, 0, 0, 1, 0],
    'UK-874': [0, 0, 0, 0, 1]
}

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    features = request.form.to_dict()
    
    # I convert input values to integers
    for key, value in features.items():
        if key not in ('airline', 'flight'):
            features[key] = int(value)
    
    # I handle one-hot encoded airline variable
    selected_airline = airline_options.get(features['airline'], [0, 0, 0, 0, 0])
    
    # I handle one-hot encoded flight variable
    selected_flight = flight_options.get(features['flight'], [0, 0, 0, 0, 0])

    final_features = [
        features['stops'],
        features['class'],
        features['duration'],
        features['days_left']
    ]

    final_features.extend(selected_airline)
    final_features.extend(selected_flight)
    final_features.append(features['arrival_time'])
    final_features.append(features['departure_time'])

    # Reshape the features to match the model's input shape (1, num_features)
    final_features = np.array(final_features).reshape(1, -1)

    prediction = model.predict(final_features)  
    output = round(prediction[0], 2)

    # Return the prediction to be displayed on the HTML page
    return render_template('index.html', prediction_text=f'Predicted Flight Price: {output}')

if __name__ == '__main__':
    app.run(debug=True)
