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


# loading the trained model
pickle_in = open('./LightGBMClassifier_all.pkl', 'rb') 
classifier = pickle.load(pickle_in)


def prediction(df):
    prediction = classifier.predict_proba(df)[0]   # A et B (A = 1 - B)
    return prediction


@st.cache(allow_output_mutation=True)
def load_data(f_csv, nrows=None, comp='infer'):
    df_f = pd.read_csv(f_csv, nrows=nrows, compression=comp)
    numeric_cols = df_f.select_dtypes(['float','int']).columns
    text_cols = df_f.select_dtypes(['object']).columns
    return df_f, numeric_cols, text_cols


def main():
    MLFLOW_URI = 'http://127.0.0.1:5000/invocations'

    # Chargement des données (remplacement des valeurs NaN par 0)
    data_load_state = st.text('Chargement des données...')
    data, n1_cols, t1_cols = load_data('./test_new_all.csv.zip', comp='zip')
    #data, n1_cols, t1_cols = load_data('./test_new_all.csv')
    data.fillna(0, inplace=True)
    df, n2_cols, t2_cols = load_data('./sub_model_B_all.csv')  #, nrows=x)
    df.fillna(0, inplace=True)
    data_load_state.text("Done! (... avec les fichiers chargés en cache)")
    df.fillna(0, inplace=True)
    feat_6 = df.iloc[:,2:].columns
    #st.write(feat_6)    
 
    # Titre du dashboard
    st.write("Version de STREAMLIT: " + st.__version__)
    # aspect Front end 
    html_titre = """ 
        <div style ="background-color:white;padding:13px"> 
        <h1 style ="color:blue;text-align:center;">Streamlit Dashboard Scoring Credit</h1> 
        </div> 
        """
    #st.title('Dashboard Scoring Credit')
    st.markdown(html_titre, unsafe_allow_html = True) 


# Sidebar

    # Choix du seuil
    st.sidebar.subheader("Seuil d'acceptation des crédits")
    seuil = st.sidebar.slider("Choisir le seuil", min_value=0.5, max_value=1.0,
        value=0.75, step=0.01)
    #st.sidebar.subheader("Choix des fichiers")
    #uploaded_file = st.sidebar.file_uploader(
    #        label="Charger votre fichier CSV ou Excel (200~MB max)", type=['csv', 'xlsx'])

    # checkbox widget
    checkbox = st.sidebar.checkbox(label="Montrer les'histogramme des données")
    if checkbox:
        st.dataframe(data=df.head(3))
        df_a = pd.DataFrame(df[:], columns=['TARGET'])
        fig, ax = plt.subplots(figsize=(5,2))
        ax.hist(df_a['TARGET'], bins=150)
        plt.show()
        st.pyplot(fig)

    st.sidebar.subheader("Choix du client")
    client = st.sidebar.selectbox(label='Liste de tous les clients',
        options=df['SK_ID_CURR'], index=1)
    df_client = df[df['SK_ID_CURR'] == client]
    df_client_v = list(df_client['TARGET'])[0]     # Formatage de la valeur
    df_client_v = "{:.3f}".format(df_client_v)

    non_accepte = st.sidebar.checkbox(label="Clients refusés")
    if non_accepte:
        st.sidebar.subheader("Choix du clients non accepté")
        df_bad = df[df['TARGET'] <= seuil]
        st.write("Nombre de clients refusés", df_bad.shape[0])
        n_index = int(df_bad.index[0])
        #st.write(df_bad.head(3)), 
        client2 = st.sidebar.selectbox(label='liste des clients refusés',
            options=df_bad['SK_ID_CURR'], index=n_index)
        st.write(f'Vous êtes le client {client2}')
        df_client2 = df[df['SK_ID_CURR'] == client2]
        df_client_v2 = list(df_client2['TARGET'])[0]     # Formatage de la valeur
        df_client_v2 = "{:.3f}".format(df_client_v2)
    else:
        st.write(f'Vous êtes le client {client}')

    # Les 6 variables
    slide0 = [0]*len(feat_6)
    slide = [0]*len(feat_6)
    features = {}
    features_new = {}
    for n, var in enumerate(feat_6):
        slide0[n] = list(df_client[var].values)[0]
        factor = 100
        maxi = int(np.max(df[var]))
        if maxi != 0:
            factor = 1
        #st.write(n, var, int(factor*slide0[n]), factor*(np.max(df[var])))
        slide[n] = st.sidebar.slider("Variable: {}".format(var),
            min_value=factor*int(np.min(df[var])), max_value=factor*int(np.max(df[var])+1),
            value=int(factor*slide0[n]), step=1)

        features[var] = slide0[n]
        features_new[var] = slide[n]
    features_df = pd.DataFrame([features])
    st.table(features_df)

# Fenetre principale  et  Nouvelles valeurs
    if non_accepte:
        proba = df_client_v2
        donnees = data[data['SK_ID_CURR'] == client2]
    else:
        proba = df_client_v
        donnees = data[data['SK_ID_CURR'] == client]
    st.write(f'Votre probabilité de remboursement est {proba}')
    if float(proba) >= seuil:   # 0.75
        st.success("Credit ...  Accordé ")
    else:
        st.error("Credit ...  Refusé ")

 
    # Suppression de certaines colonnes:
    # ['TARGET','SK_ID_CURR','SK_ID_BUREAU','SK_ID_PREV','index', 'TARGET', 'PREDICTIONS']
    donnees.drop(['SK_ID_CURR'], axis=1, inplace=True)
    #st.write(donnees.shape)

    for n, var in enumerate(feat_6):
        factor = 100
        maxi = int(np.max(df[var]))
        if maxi != 0:
            factor = 1
        #st.write(var, "  Old:",int(factor*slide0[n]), "New",slide[n])
        #st.write(n, var, list(donnees[var].values)[0], slide[n])
        donnees[var] = slide[n]

    features_new_df = pd.DataFrame([features_new])
    st.table(features_new_df)


    if st.button('Recalculer'):
        st.write(donnees)
        pred = None
        #if api_choice == 'MLflow':
        pred = prediction(donnees)[0]
        #pred = request_prediction(MLFLOW_URI, donnees)[0]
        st.write('La nouvelle probabilité est {:.3f}'.format(pred))


if __name__ == '__main__':
    main()
