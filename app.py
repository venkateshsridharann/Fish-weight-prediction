import os
from flask import Flask, request, jsonify
import pickle

app = Flask(__name__)

# Load the model
with open('model/fish_weight_predictor.pkl', 'rb') as f:
    model = pickle.load(f)

@app.route('/')
def home():
    return "Fish Weight Predictor"

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    prediction = model.predict([[
        data['Length1'],
        data['Length2'],
        data['Length3'],
        data['Height'],
        data['Width']
    ]])
    return jsonify({'prediction': prediction[0]})

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=True, host='0.0.0.0', port=port)
