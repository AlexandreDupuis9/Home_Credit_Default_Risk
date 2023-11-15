import numpy as np
import pandas as pd
import streamlit as st
import requests
import matplotlib.pyplot as plt
import shap as shap

client_data = None
client_id_list = None
list_features = None


def load_data():
    global client_data, client_id_list, list_features
    # Appeler l'API pour récupérer la liste des identifiants clients
    response = requests.get('http://127.0.0.1:5000/client_data')
    if response.status_code == 200:
        client_data = response.json(allow_nan=True)['client_data']
        client_id_list = [entry['SK_ID_CURR'] for entry in client_data]
        client_data = pd.DataFrame(client_data)
    else:
        st.error(f"Error fetching client data. Status code: {response.status_code}")

    response = requests.get('http://127.0.0.1:5000/get_list_features')
    list_features = response.json()['list_features']


def main():
    st.title("Customer Relationship Manager Dashboard")

    if not client_data or not client_id_list or not list_features:
        load_data()

    st.sidebar.title("Select Customer")
    selected_customer = st.sidebar.selectbox("Choose a customer:", client_id_list)
    customer_info = client_data[client_data["SK_ID_CURR"] == selected_customer]
    st.write(customer_info)

    # Appeler l'API pour récupérer les informations du client sélectionné
    response = requests.post('http://127.0.0.1:5000/predict', json={'features': {'SK_ID_CURR': selected_customer}})
    prediction = response.json()['prediction']

    st.subheader("Customer Information")
    st.write(f"Probability of Positive Response: {prediction:.2f}")

    selected_feature = st.selectbox("Select feature for SHAP visualization:", list_features)

    response = requests.post('http://127.0.0.1:5000/get_shap', json={'features': {'SK_ID_CURR': selected_customer}})
    shap_values = response.json()['shap_values']

    # Obtenez la distribution de ce critère pour tous les clients
    all_clients_values = client_data[selected_feature].values

    # Divisez l'espace en quatre colonnes
    col1, col2 = st.columns(2)

    # Affichez chaque graphique dans une colonne différente
    with col1:
        # Créez un graphique pour comparer
        fig, ax = plt.subplots()
        ax.hist(all_clients_values, bins=20, alpha=0.5, color='b', label='All Clients')
        ax.axvline(x=selected_feature, color='r', linestyle='dashed', linewidth=2, label='Selected Client')
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

    array = np.array(shap_values)

    #st.subheader(f"SHAP Global Values")
    #shap.summary_plot(array[0], list_features, feature_names=list_features, plot_type="bar", show=False)
    #st.pyplot()


if __name__ == "__main__":
    main()
