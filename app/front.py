import numpy as np
import pandas as pd
import streamlit as st
import requests
import matplotlib.pyplot as plt
import streamlit as st
import plotly.graph_objects as go
import shap as shap

client_data = None
client_id_list = None
list_features = None

# Réaliser un dashboard pour présenter son travail de modélisation
    # CE1 Vous avez décrit et conçu un parcours utilisateur simple permettant de répondre aux besoins des utilisateurs (les différentes actions et clics sur les différents graphiques permettant de répondre à une question que se pose l'utilisateur).
    # CE2 Vous avez développé au moins deux graphiques interactifs permettant aux utilisateurs d'explorer les données.
    # CE3 Vous avez réalisé des graphiques lisibles (taille de texte suffisante, définition lisible).
    # CE4 Vous avez réalisé des graphiques qui permettent de répondre à la problématique métier.
    # CE5 Vous avez pris en compte le besoin des personnes en situation de handicap dans la réalisation des graphiques : le candidat doit avoir pris en compte au minimum les critères d'accessibilité du WCAG suivants (https://www.w3.org/Translations/WCAG21-fr/):
        # Critère de succès 1.1.1 Contenu non textuel
        # Critère de succès 1.4.1 Utilisation de la couleur
        # Critère de succès 1.4.3 Contraste (minimum)
        # Critère de succès 1.4.4 Redimensionnement du texte
        # Critère de succès 2.4.2 Titre de page
    # CE6 Vous avez déployé le dashboard sur le web afin qu'il soit accessible pour d'autres utilisateurs sur leurs postes de travail.

def load_data():
    global client_data, client_id_list, list_features
    # Appeler l'API pour récupérer la liste des identifiants clients
    response = requests.get('http://127.0.0.1:5000/client_data')
    if response.status_code == 200:
        client_data = response.json()['client_data']
        client_id_list = [entry['SK_ID_CURR'] for entry in client_data]
        client_data = pd.DataFrame(client_data)
    else:
        st.error(f"Error fetching client data. Status code: {response.status_code}")

    response = requests.get('http://127.0.0.1:5000/get_list_features')
    list_features = response.json()['list_features']


def main():
    st.title("Customer Risk Credit dashboard")

    if not client_data or not client_id_list or not list_features:
        load_data()

    st.sidebar.title("Select Customer")
    selected_customer = st.sidebar.selectbox("Choose a customer:", client_id_list)
    customer_info = client_data[client_data["SK_ID_CURR"] == selected_customer]
    st.subheader("Customer Information")
    st.write(customer_info)

    # Appeler l'API pour récupérer les informations du client sélectionné
    response = requests.get('http://127.0.0.1:5000/predict', json={'features': {'SK_ID_CURR': selected_customer}})
    prediction = response.json()['prediction']

    st.subheader("Customer Risk")
    colors = ["#008000", "#FFD700", "#FF0000"]  # Vert, Jaune, Rouge
    st.write("1 - client with payment difficulties: he/she had late payment more than X days on at least one of the first Y installments of the loan in our sample")
    st.write("0 - all other cases")
    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = prediction,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "Risk of payment difficulties"},
        gauge = {
            'axis': {'range': [None, 1], 'tickwidth': 1, 'tickcolor': "darkblue"},
            'bar': {'color': "darkblue"},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, 0.3], 'color': colors[0]},
                {'range': [0.3, 0.7], 'color': colors[1]},
                {'range': [0.7, 1], 'color': colors[2]}
            ],
            'threshold': {
                'line': {'color': "darkblue", 'width': 4},
                'thickness': 0.75,
                'value': prediction
            }
        }
    ))

    fig.update_layout(paper_bgcolor = "white", font = {'color': "darkblue", 'family': "Arial"})
    st.plotly_chart(fig)


    st.subheader("Comparative Analysis")
    selected_feature = st.selectbox("Select feature to focus :", list_features)
    response = requests.get('http://127.0.0.1:5000/get_column_description', json={'column_name': selected_feature})
    description = response.json()['description']
    all_clients_values = client_data[selected_feature].values

    col1, col2 = st.columns(2)
    st.write(f'{selected_feature} description : {description}')
    if pd.notna(customer_info[selected_feature].iloc[0]):
        st.write(f'{selected_feature} value : {customer_info[selected_feature].iloc[0]}')

        with col1:
            fig, ax = plt.subplots()
            ax.hist(all_clients_values, bins=50, alpha=0.5, color='b', label='All Clients')
            ax.axvline(x=customer_info[selected_feature].iloc[0], color='r', linestyle='dashed', linewidth=2, label='Selected Client')
            ax.set_xlabel(selected_feature)
            ax.set_ylabel('Frequency')
            ax.legend(loc='upper right')
            st.pyplot(fig)
    else:
        st.write(f'No available value for {selected_feature}.')

    with col2:
        #st.pyplot(fig)
        st.set_option('deprecation.showPyplotGlobalUse', False)

        fig2, ax2 = plt.subplots()
        ax2.hist(client_data["AMT_INCOME_TOTAL"].values, bins=50, alpha=0.5, color='b', label='All Clients')
        ax2.axvline(x=customer_info["AMT_INCOME_TOTAL"].values[0], color='r', linestyle='dashed', linewidth=2, label='Selected Client')
        ax2.set_xlabel("AMT_INCOME_TOTAL")
        ax2.set_ylabel('Frequency')
        ax2.legend(loc='upper right')
        st.pyplot()


    st.subheader(f"Feature Impact on Prediction")

    response = requests.get('http://127.0.0.1:5000/get_shap', json={'features': {'SK_ID_CURR': selected_customer}})
    shap_values = response.json()['shap_values']

    array = np.array(shap_values)

    st.write("This plot shows how each feature contributes to the model's predictions.")
    st.write("If the impactful feature is greater than 0 in the graph, then it will influence the risk towards 1")
    plt.figure(figsize=(15, 11))
    shap.summary_plot(array[1], list_features, feature_names=list_features, plot_type="dot", max_display=15, show=False)
    st.pyplot(plt)

if __name__ == "__main__":
    main()