import pickle

from flask import Flask, request, jsonify
import pandas as pd
import lightgbm as lgb
import shap as shap

app = Flask(__name__)

model = lgb.Booster(model_file='./model.txt')

client_data = None
client_id_list = None

def load_client_data():
    global client_data, client_id_list
    client_data = pd.read_csv('processed_data.csv', nrows=1000)
    client_id_list = client_data['SK_ID_CURR'].unique()

@app.route('/client_data', methods=['GET'])
def get_client_data():
    global client_data
    if client_data is None:
        load_client_data()
    return jsonify({'client_data': client_data.to_dict(orient='records')})

def make_predictions(data):
    predictions = model.predict(data)
    return predictions

def compute_shap_values(explainer, data):
    model.params['objective'] = 'binary'
    shap_values = explainer.shap_values(data)
    return shap_values

@app.route('/get_list_features', methods=['GET'])
def get_list_features():
    return jsonify({'list_features': model.feature_name()})

@app.route('/predict', methods=['POST'])
def predict():

    global client_data
    if client_data is None:
        load_client_data()

    selected_customer = request.json['features']['SK_ID_CURR']
    selected_customer_data = client_data[client_data["SK_ID_CURR"] == selected_customer]

    list_of_features = model.feature_name()

    print(list_of_features)

    prediction = make_predictions(selected_customer_data[list_of_features])

    return jsonify({'prediction': float(prediction[0])})


# Nouvelle route pour récupérer le modèle
@app.route('/get_shap', methods=['POST'])
def get_shap():

    global client_data
    if client_data is None:
        load_client_data()

    selected_customer = request.json['features']['SK_ID_CURR']
    selected_customer_data = client_data[client_data["SK_ID_CURR"] == selected_customer]
    explainer = shap.TreeExplainer(model)
    shap_values = compute_shap_values(explainer, selected_customer_data[model.feature_name()])

    shap_values_list = [arr.tolist() for arr in shap_values]

    return jsonify({'shap_values': shap_values_list})

if __name__ == '__main__':
    load_client_data()
    app.run(debug=True)