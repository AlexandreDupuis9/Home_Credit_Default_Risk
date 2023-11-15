import streamlit as st
import pandas as pd
import lightgbm as lgb
import shap as shap
import matplotlib.pyplot as plt

# Load your trained model here (replace 'path_to_model' with the actual path)
model = lgb.Booster(model_file='../notebook/model.txt')

# Load customer data (replace 'path_to_customer_data' with the actual path)
client_data = pd.read_csv('../ressources/processed_data.csv', nrows=1000)
client_id_list = client_data['SK_ID_CURR'].unique()
#specific_client_data = client_data[client_data['SK_ID_CURR'] == client_id]



# Permettre de visualiser le score et l’interprétation de ce score pour chaque client de façon intelligible pour une personne non experte en data science. pip install streamlit-echarts + SHAP
# Permettre de visualiser des informations descriptives relatives à un client (via un système de filtre).
# Permettre de comparer les informations descriptives relatives à un client à l’ensemble des clients ou à un groupe de clients similaires.
#https://towardsdatascience.com/how-to-easily-deploy-machine-learning-models-using-flask-b95af8fe34d4

def make_predictions(data):
    predictions = model.predict(data)
    return predictions

# Function to compute SHAP values
def compute_shap_values(explainer, data):
    model.params['objective'] = 'binary'
    shap_values = explainer.shap_values(data)
    return shap_values

def visualize_shap_chart(shap_values, feature_names):
    shap_values_2d = shap_values[0].tolist()

    shap_df = pd.DataFrame(shap_values_2d, columns=feature_names)

    selected_feature = st.selectbox("Select feature for SHAP visualization:", feature_names)
    chart_data = shap_df[selected_feature]

    options = {
        "xAxis": {"type": "category", "data": list(shap_df.index)},
        "yAxis": {"type": "value"},
        "series": [{"data": chart_data, "type": "bar"}],
        "title": {"text": f"SHAP Values for {selected_feature}"}
    }
    st(options=options)

def main():
    st.title("Customer Relationship Manager Dashboard")

    st.sidebar.title("Select Customer")
    selected_customer = st.sidebar.selectbox("Choose a customer:", client_id_list)


    st.subheader("Customer Information")
    customer_info = client_data[client_data["SK_ID_CURR"] == selected_customer]
    st.write(customer_info)

    st.subheader("Model Predictions")
    #selected_customer_data = customer_info.drop(columns=["SK_ID_CURR"])
    feats = [f for f in customer_info.columns if f not in ['TARGET','SK_ID_CURR','SK_ID_BUREAU','SK_ID_PREV','index']]

    list_of_features = model.feature_name()

    print(list_of_features)

    prediction = make_predictions(customer_info[list_of_features])
    st.write(f"Probability of Positive Response: {prediction[0]:.2f}")

    explainer = shap.TreeExplainer(model)
    shap_values = compute_shap_values(explainer, customer_info[list_of_features])




    selected_feature = st.selectbox("Select feature for SHAP visualization:", list_of_features)

    feature_index = list_of_features.index(selected_feature)

    # Obtenez la valeur du critère pour le client sélectionné
    selected_customer_value = customer_info[selected_feature].values[0]

    # Obtenez la distribution de ce critère pour tous les clients
    all_clients_values = client_data[selected_feature].values

    # Divisez l'espace en quatre colonnes
    col1, col2 = st.columns(2)

    # Affichez chaque graphique dans une colonne différente
    with col1:
            # Créez un graphique pour comparer
            fig, ax = plt.subplots()
            ax.hist(all_clients_values, bins=20, alpha=0.5, color='b', label='All Clients')
            ax.axvline(x=selected_customer_value, color='r', linestyle='dashed', linewidth=2, label='Selected Client')
            ax.set_xlabel(selected_feature)
            ax.set_ylabel('Frequency')
            ax.legend(loc='upper right')
            st.pyplot(fig)

    with col2:
            # Affichez le graphique avec Streamlit
            #st.pyplot(fig)
            st.set_option('deprecation.showPyplotGlobalUse', False)

            # Créez un graphique pour comparer
            fig2, ax2 = plt.subplots()
            ax2.hist(client_data["AMT_INCOME_TOTAL"].values, bins=20, alpha=0.5, color='b', label='All Clients')
            ax2.axvline(x=customer_info["AMT_INCOME_TOTAL"].values[0], color='r', linestyle='dashed', linewidth=2, label='Selected Client')
            ax2.set_xlabel("AMT_INCOME_TOTAL")
            ax2.set_ylabel('Frequency')
            ax2.legend(loc='upper right')
            # Affichez le graphique avec Streamlit
            st.pyplot()

    st.subheader(f"SHAP Global Values")
    shap.summary_plot(shap_values, list_of_features, feature_names=list_of_features, plot_type="bar", show=False)
    st.pyplot()


if __name__ == "__main__":
    main()
#%%
