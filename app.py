from flask import Flask, request, jsonify, render_template
import pickle
import pandas as pd
import os

app = Flask(__name__)

# Load the trained model
with open('./static/fish_weight_predictor.pkl', 'rb') as f:
    model = pickle.load(f)
    
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.form.to_dict()
    df = pd.DataFrame([data])
    df['Species'] = df['Species'].astype('category').cat.codes
    prediction = model.predict(df)
    return jsonify({'prediction': prediction[0]})

if __name__ == '__main__':
    # Get the port from the environment variable and set a default if not found
    port = int(os.environ.get('PORT', 5000))
    # Run the app, listening on all interfaces on the specified port
    app.run(host='0.0.0.0', port=port, debug=True)
