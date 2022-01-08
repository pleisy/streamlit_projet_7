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


#@st.cache(allow_output_mutation=True)
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
 
    # Title of the dashboard
    st.write("Version de STREAMLIT: " + st.__version__)
    st.title('Dashboard Scoring Credit')

# Sidebar
    st.sidebar.subheader("Choix des fichiers")
    uploaded_file = st.sidebar.file_uploader(
            label="Charger votre fichier CSV ou Excel (200~MB max)", type=['csv', 'xlsx'])

    # checkbox widget
    checkbox = st.sidebar.checkbox(label="Montrer les données")
    if checkbox:
        st.dataframe(data=df.head(3))
        df_a = pd.DataFrame(df[:], columns=['TARGET'])
        fig, ax = plt.subplots(figsize=(5,2))
        ax.hist(df_a['TARGET'], bins=150)
        plt.show()
        st.pyplot(fig)

    st.sidebar.subheader("Choix du client")
    client = st.sidebar.selectbox(label='', options=df['SK_ID_CURR'], index=0)
    st.write(f'Vous êtes le client {client}')
    df_client = df[df['SK_ID_CURR'] == client]
    df_client_v = list(df_client['TARGET'])[0]     # Formatage de la valeur
    df_client_v = "{:.3f}".format(df_client_v)


    if st.sidebar.checkbox(label="Mauvais clients"):
        st.sidebar.subheader("Clients non acceptés")
        df_bad = df[df['TARGET'] <= 0.80]
        st.write("Nombre de BAD clients", df_bad.shape)
        st.write(df_bad.head(3))
        client2 = st.sidebar.selectbox(label='', options=df['TARGET'], index=1)
        st.write(f'Vous êtes le client {client2}')
        df_client2 = df[df['TARGET'] == client2]
        df_client_v2 = list(df_client2['TARGET'])[0]     # Formatage de la valeur
        df_client_v2 = "{:.3f}".format(df_client_v2)


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
        slide[n] = st.sidebar.slider("Variable {}".format(var),
            min_value=factor*int(np.min(df[var])), max_value=factor*int(np.max(df[var])+1),
            value=int(factor*slide0[n]), step=1)

        features[var] = slide0[n]
        features_new[var] = slide[n]
    features_df = pd.DataFrame([features])
    st.table(features_df)

# Fenetre principale
    st.write(f'Votre probabilité de remboursement est {df_client_v}')
    if float(df_client_v) >= 0.85:   # 0.75
        st.success("Credit ...  Accordé ")
    else:
        st.error("Credit ...  Refusé ")


    # Nouvelles valeurs
    donnees = data[data['SK_ID_CURR'] == client]
    # Suppression de certaines colonnes:
    # ['TARGET','SK_ID_CURR','SK_ID_BUREAU','SK_ID_PREV','index', 'TARGET', 'PREDICTIONS']
    donnees.drop(['SK_ID_CURR'], axis=1, inplace=True)
    st.write(donnees.shape)

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
