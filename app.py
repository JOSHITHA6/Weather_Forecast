from flask import Flask, request, jsonify
import joblib

# Initialize Flask app
app = Flask(__name__)

# Load the pre-trained model (you should have a model saved using joblib)
model = joblib.load('Gradient Boosting_90_9.pkl')  # Replace with your model's filename

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    dewPoint = data['dewPoint']
    humidity = data['humidity']
    windSpeed = data['windSpeed']
    visibility = data['visibility']
    pressure = data['pressure']
    target = data['target']

    # Prepare the input data for prediction
    input_data = [[dewPoint, humidity, windSpeed, visibility, pressure]]

    # Make the prediction using your model
    if target == 'temperature':
        prediction = model.predict(input_data)  # Assuming your model predicts temperature
    elif target == 'visibility':
        prediction = model.predict(input_data)  # Assuming your model predicts visibility

    # Send the result back
    return jsonify({'prediction': prediction[0]})

if __name__ == '__main__':
    app.run(debug=True)
