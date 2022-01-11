import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import requests
import pickle
from lightgbm import LGBMClassifier

# from lightgbm import LGBMClassifier

# set the style for seaborn
sns.set_style('darkgrid')
# Suppression des messages de deprecation
st.set_option('deprecation.showfileUploaderEncoding', False)


def request_prediction(model_uri, data):
    headers = {"Content-Type": "application/json"}

    data_j = data.to_json
    #st.write(data_j)
    data_json = {'data': data_j}
    response = requests.request(
        method='POST', headers=headers, url=model_uri, json=data_json)

    if response.status_code != 200:
        raise Exception(
            "Request failed with status {}, {}".format(response.status_code, response.text))

    return response.json()


# Chargement du model entrainé
pickle_in = open('./LightGBMClassifier_all.pkl', 'rb') 
classifier = pickle.load(pickle_in)


def prediction(df):
    prediction = classifier.predict_proba(df)[0]   # A et B (A = 1 - B)
    return prediction


def pret_accord(texte, proba, seuil):
    #st.write(str(texte) + '{:.3f}'.format(str(proba)))
    st.write('La probabilité de remboursement est :', proba)
    if float(proba) >= seuil:
        st.success("Credit ...  Accordé ")
    else:
        st.error("Credit ...  Refusé ")


@st.cache(allow_output_mutation=True)
def load_data(f_csv, nrows=None, comp='infer'):
    df_f = pd.read_csv(f_csv, nrows=nrows, compression=comp)
    numeric_cols = df_f.select_dtypes(['float','int']).columns
    text_cols = df_f.select_dtypes(['object']).columns
    return df_f, numeric_cols, text_cols


def main():
    #MLFLOW_URI = 'http://127.0.0.1:5000/invocations'

    # Chargement des données (remplacement des valeurs NaN par 0)
    data_load_state = st.text('Chargement des données...')
    data, n1_cols, t1_cols = load_data('./test_new_all.csv.zip', comp='zip')
    #data, n1_cols, t1_cols = load_data('./test_new_all.csv')
    data.fillna(0, inplace=True)
    df, n2_cols, t2_cols = load_data('./sub_model_B_all.csv')  #, nrows=x)
    df.fillna(0, inplace=True)
    data_load_state.text("Done! (... fichiers sont chargés en cache)")
    df.fillna(0, inplace=True)
    feat_6 = df.iloc[:,2:].columns      #st.write(feat_6)    
 
    # Titre du dashboard
    st.write("Version de STREAMLIT: " + st.__version__)
    # aspect Front end 
    html_titre = """ 
        <div style ="background-color:yellow;padding:13px"> 
        <h1 style ="color:blue;text-align:center;">STREAMLIT: Dashboard Scoring Credit</h1> 
        </div> 
        """
    #st.title('Dashboard Scoring Credit')
    st.markdown(html_titre, unsafe_allow_html = True) 


# Sidebar
    # Choix du seuil
    st.sidebar.subheader("Seuil d'acceptation des crédits")
    seuil = st.sidebar.slider("Choisir le seuil", min_value=0.20, max_value=1.0,
        value=0.75, step=0.01)
    #st.sidebar.subheader("Choix des fichiers")
    #uploaded_file = st.sidebar.file_uploader(
    #        label="Charger votre fichier CSV ou Excel (200~MB max)", type=['csv', 'xlsx'])
    # checkbox widget
    boxHisto = st.sidebar.checkbox(label="Montrer les'histogramme des données")
    if boxHisto:
        #st.dataframe(data=df.head(3))
        df_a = pd.DataFrame(df[:], columns=['TARGET'])
        fig, ax = plt.subplots(figsize=(5,2))
        ax.hist(df_a['TARGET'], bins=150)
        plt.show()
        st.pyplot(fig)

    # Label du client choisit
    non_accepte = st.sidebar.checkbox(label="Clients refusés")
    if non_accepte:
        st.sidebar.subheader("Choix du client non accepté")
        # REST_INDEX ... correction sinon BUG index trop grands / nombre de valeurs
        df_bad = df[df['TARGET'] <= seuil].reset_index()
        if df_bad.shape[0] > 0:
            st.write("Nombre de clients refusés", df_bad.shape[0])
            n_index = int(df_bad.index[0])
            client = st.sidebar.selectbox(label='liste des clients refusés',
                options=df_bad['SK_ID_CURR'], index=n_index)
            df_client = df_bad[df_bad['SK_ID_CURR'] == client]
        else:
            st.error("Aucun client refusé")
    else:
        st.sidebar.subheader("Choix du client")
        client = st.sidebar.selectbox(label='Liste de tous les clients',
            options=df['SK_ID_CURR'], index=0)
        df_client = df[df['SK_ID_CURR'] == client]
 
    st.write(f'Vous êtes le client {client}')
    df_client_val = list(df_client['TARGET'])[0]     # Formatage de la valeur
    df_client_val = "{:.3f}".format(df_client_val)

    # Les 6 variables
    slide0 = [0]*len(feat_6)
    slide = [0]*len(feat_6)
    features = {}
    features_new = {}
    for n, var in enumerate(feat_6):
        slide0[n] = list(df_client[var].values)[0]
        if non_accepte:
            mini = float(np.min(df_bad[var]))
            maxi = float(np.max(df_bad[var]))
            slide[n] = st.sidebar.slider("Variable: {}".format(var),
                min_value=mini, max_value=maxi, value=float(slide0[n]), step=1/100)
        else:
            mini = float(np.min(df[var]))
            maxi = float(np.max(df[var]))
            slide[n] = st.sidebar.slider("Variable: {}".format(var),
                min_value=mini, max_value=maxi, value=float(slide0[n]), step=1/100)

        features[var] = slide0[n]
        features_new[var] = slide[n]
    features_df = pd.DataFrame([features])
    st.table(features_df)
    features_new_df = pd.DataFrame([features_new])
    st.table(features_new_df)

# Fenetre principale  et  Nouvelles valeurs
# Suppression de certaines colonnes:
# ['TARGET','SK_ID_CURR','SK_ID_BUREAU','SK_ID_PREV','index', 'TARGET', 'PREDICTIONS']  
 
    proba = df_client_val
 #   if non_accepte:
 #       data_bad = data[data['TARGET'] <= seuil].reset_index()
#        donnees = data_bad[data_bad['SK_ID_CURR'] == client]
#    else:
    donnees = data[data['SK_ID_CURR'] == client].reset_index()
    donnees.drop(['SK_ID_CURR', 'index'], axis=1, inplace=True)
    features_new = {}
    for n, var in enumerate(feat_6):
        donnees[var] = slide[n]
        
    titre = "Votre probabilité de remboursement est: " + proba
    pret_accord(titre, proba, seuil)
    st.write(donnees)

    if st.button('Recalculer'):
        #st.write(donnees)
        pred = None
        #if api_choice == 'MLflow':
        pred = "{:.3f}".format(prediction(donnees)[0])
        #pred = request_prediction(MLFLOW_URI, donnees)[0]
        titre = "La nouvelle probabilité de remboursement est: " + pred
        pret_accord(titre, pred, seuil)

if __name__ == '__main__':
    main()
