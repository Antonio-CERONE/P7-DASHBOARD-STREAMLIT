import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pandas import json_normalize
import plotly.express as px
import seaborn as sns
import shap
import streamlit as st
import streamlit.components.v1 as components
import requests
import json

# Constantes
threshold = 0.422
days_to_year = 365


def main():
    @st.cache
    def charger_donnees():
        data = pd.read_csv('data_sample_target.csv')
        data.set_index("SK_ID_CURR", inplace=True)
        data.sort_index(ascending=True, inplace=True)
        target = data["TARGET"]
        data_client = data.index
        data_age = round(-1 * (data["DAYS_BIRTH"] // days_to_year), 0)
        data_income = round(data["AMT_INCOME_TOTAL"], 0)
        targets = data.TARGET.value_counts()  # pour Pie Chart

        description = pd.read_csv('HomeCredit_columns_description.csv',
                                  usecols=['Row', 'Description'],
                                  index_col=0,
                                  encoding='utf-8')

        lst_infos = [data.shape[0],
                     round(data["AMT_INCOME_TOTAL"].mean(), 0),
                     round(data["AMT_CREDIT"].mean(), 0)]
        nb_credits = lst_infos[0]
        rev_moy = lst_infos[1]
        credits_moy = lst_infos[2]

        clf = joblib.load("lgbm_model_saved.joblib")

        return data, target, data_client, data_age, data_income, targets, description, lst_infos, nb_credits, \
               rev_moy, credits_moy, clf

    @st.cache
    def identite_client_target(data, id):
        data_client_target = data[data.index == int(id)]
        return data_client_target

    def identite_client(data, id):
        data_without_target = data[data.index == int(id)].drop(columns="TARGET")
        return data_without_target

    @st.cache
    def load_prediction(clf, id):
        score = clf.predict_proba(identite_client(id))
        return clf, score

    def st_shap(plot, height=None):
        shap_html = f"<head>{shap.getjs()}</head><body>{plot.html()}</body>"
        components.html(shap_html, height=height)

    # Charger données
    data, target, data_client, data_age, data_income, targets, description, lst_infos, nb_credits, \
    rev_moy, credits_moy, clf = charger_donnees()

    # Title #
    #########
    html_temp = """
            <div style="background-color: #992717; padding:10px; border-radius:48px">
            <h1 style="color: white; text-align:center">Credit Scoring - Dashboard</h1>
            </div>
            <p style="font-size: 16px; font-weight: bold; text-align:center">Aide à la décision d'octroi-crédit</p>
            """
    st.markdown(html_temp, unsafe_allow_html=True)

    # Left SideBar #
    ################
    # Choix du Client
    st.sidebar.header("**Information Générale**")

    # Charger la selectbox
    chk_id = st.sidebar.selectbox("Client ID", data_client, index=0)

    # Charger les informations générales
    data, target, data_client, data_age, data_income, targets, description, lst_infos, nb_credits, \
    rev_moy, credits_moy, clf = charger_donnees()

    # Nombre de crédits dans l'échantillon
    st.sidebar.markdown("<u>Nb de demandes de crédits à visualiser</u>", unsafe_allow_html=True)
    st.sidebar.text(nb_credits)

    # Revenus annuels moyens
    st.sidebar.markdown("<u>Revenus annuels moyens</u>\n(USD)", unsafe_allow_html=True)
    st.sidebar.text(rev_moy)

    # Montant moyen des credits à la consommation
    st.sidebar.markdown("<u>Montant moyen des crédits</u>\n(USD)", unsafe_allow_html=True)
    st.sidebar.text(credits_moy)

    # PieChart
    fig, ax = plt.subplots(figsize=(5, 5))
    plt.pie(targets,
            explode=[0, 0.05],
            labels=['Sans Défaut de Paiement\n(Classe 0)', 'Défaut de Paiement\n(Classe 1)'],
            autopct='%1.1f%%',
            startangle=120)
    st.sidebar.pyplot(fig)

    # HOME PAGE - MAIN CONTENT #
    ############################
    # Affichage des informations générales du Client choisi dans selectbox  : Sexe, Age, Etat Civil, Composition du foyer, …
    st.header("**Affichage des informations générales Client**")
    # Display Customer ID from Sidebar
    st.write("Client selectionné :", chk_id)

    if st.checkbox("Voulez-vous voir ses informations ?", value=False):
        data_client = identite_client(data, chk_id)
        st.write("Sexe : ", data_client["CODE_GENDER"].values[0])
        st.write("Age : {:.0f} ans".format(-1 * int(data_client["DAYS_BIRTH"] // days_to_year)))
        st.write("Etat civil : ", data_client["NAME_FAMILY_STATUS"].values[0])
        st.write("Nombre de personnes dans le foyer : {:.0f}".format(data_client["CNT_FAM_MEMBERS"].values[0]))

        # Graphique ; Distribution de l'âge
        st.subheader("*Distribution de l'âge moyen des Clients*")
        age_client_choisi = data_age[data_age.index == chk_id].values[0]
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.histplot(data_age, edgecolor='k', color="#B880EE", bins=15)
        ax.axvline(x=age_client_choisi, color="#2E9942", linestyle='--')
        ax.set(title=' ', xlabel='Age (ans)', ylabel="Nbr d'occurrences")
        st.pyplot(fig)

        # Graphique ; Distribution des Revenus Annuels
        st.subheader("*Distribution des Revenus Annuels des Clients*")
        revenus_client_choisi = data_income[data_income.index == chk_id].values[0]
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.histplot(data_income, edgecolor='k', color="#B880EE", bins=15)
        ax.axvline(x=revenus_client_choisi, color="#2E9942", linestyle='--')
        ax.set(title=' ', xlabel='Age (ans)', ylabel="Nbr d'occurrences")
        st.pyplot(fig)

        # Graphique ; Scatter Plot - Relations  Age vs. Revenus Annuels
        st.subheader("*Relations  Age vs. Revenus Annuels*")
        data_sk = data.reset_index(drop=False)
        data_sk.DAYS_BIRTH = (-1 * data_sk['DAYS_BIRTH'] // days_to_year).round(0)
        fig = px.scatter(data_sk,
                         x='DAYS_BIRTH',
                         y="AMT_INCOME_TOTAL",
                         size="AMT_INCOME_TOTAL",
                         color='CODE_GENDER',
                         color_discrete_sequence=["pink", "lightblue"],
                         template="simple_white",
                         hover_data=['SK_ID_CURR', 'NAME_FAMILY_STATUS', 'CNT_FAM_MEMBERS', 'NAME_CONTRACT_TYPE'])

        fig.update_xaxes(showline=True,
                         linewidth=2,
                         title="Age",
                         title_font=dict(size=12, family='Verdana'))

        fig.update_yaxes(showline=True,
                         tickprefix="$",
                         linewidth=2,
                         title="Revenus Annuels (USD)",
                         title_font=dict(size=12, family='Verdana'))

        fig.update_layout({'plot_bgcolor': 'white'},
                          legend=dict(y=0.2, orientation='v'))

        fig.add_traces(px.scatter(data_sk[data_sk.SK_ID_CURR == int(chk_id)],
                                  x='DAYS_BIRTH',
                                  y="AMT_INCOME_TOTAL",
                                  color='CODE_GENDER',
                                  hover_data=['SK_ID_CURR',
                                              'NAME_FAMILY_STATUS',
                                              'CNT_FAM_MEMBERS',
                                              'NAME_CONTRACT_TYPE']).update_traces(marker_size=20,
                                                                                   marker_color="#2E9942").data)
        st.plotly_chart(fig)
    else:
        st.markdown("___", unsafe_allow_html=True)

    # Affichage des informations financières du Client
    st.header("**Analyse des données financières du Client**")
    st.markdown("### <u>Prédiction du modèle</u>", unsafe_allow_html=True)

    # Récupérer la valeur de la prédiction depuis API FLASK hebergée sur le site deployé via HEROKU
    url = "https://p7-api-flask.herokuapp.com/api_proba/client_choisi?id=" + str(chk_id)
    # url = "http://127.0.0.1:5000/api_proba/client_choisi?id=" + str(chk_id)
    reponse = requests.get(url, timeout=8)
    contenu = reponse.json()
    dict_json = json.loads(contenu)
    prediction_classe0 = round(json_normalize(dict_json["0"]).iat[0, 0], 3)
    prediction_classe1 = round(1 - prediction_classe0, 3)
    # st.write("Prediction Class 1 de l'API FLASK", prediction_classe1)

    # Récupérer la valeur de prédiction par modèle Light GBM
    prediction = clf.predict_proba(identite_client(data, chk_id))
    predict_1 = prediction[0, 1]
    target_1 = target[target.index == chk_id].values[0]
    classe_thresh = np.where(predict_1 < threshold, 0, 1).tolist()

    if target_1 == 0:
        loan = '(Client non défaillant ; Crédit remboursable)'
    else:
        loan = '(Client défaillant : Crédit non remboursable)'

    classe_thresh = np.where(predict_1 < threshold, 0, 1)
    if classe_thresh == 1:
        grant = "Désolé, le modèle préconise de ne pas vous octroyer ce crédit..."
    else:
        grant = "**Félicitations**, le modèle valide l'octroi de votre crédit !"

    depassement_classe_1 = threshold - predict_1
    if depassement_classe_1 >= 0:
        depassement = 0
    else:
        depassement = abs(depassement_classe_1)

    pourcentage_defaillance="{:.2%}".format(round(predict_1, 3))
    pourcentage_defaillance_API = "{:.2%}".format(prediction_classe1)
    pourcentage_threshold="{:.2%}".format(round(threshold, 3))

    # st.write('- Classe Réelle du Client :', target_1, loan)
    st.write('- Probabilité Prédite de Défaillance-Client : ', pourcentage_defaillance)
    st.write("- Probabilité Prédite de Défaillance-Client ; Source API FLASK : ", pourcentage_defaillance_API)
    st.write('- Seuil de décision à ne pas dépasser (i.e. threshold) : ', pourcentage_threshold)
    st.write('- Dépassement (points) : ', round(depassement*100, 2))
    st.write('- *REPONSE DU MODELE* :', grant)

    # Interprétabilité : SHAP
    shap.initjs()
    st.markdown("### <u>Interprétation des Variables</u>", unsafe_allow_html=True)
    logit_thershold = np.log(threshold / (1 - threshold))

    with open('feature_names.pkl', 'rb') as f:
        feature_names_loaded = joblib.load(f)
    feature_names = feature_names_loaded

    # ohe
    scaling = clf.named_steps['preparation']

    # standard scaler
    clf_ = clf.named_steps['clf']
    X_train_scaled = scaling.transform(data.drop(columns="TARGET"))

    # SHAP Value au format logit
    explainerlgbmc = shap.TreeExplainer(clf_)
    shap_values_train = explainerlgbmc.shap_values(X_train_scaled)

    ## GLOBAL
    # Graphique ; Interprétation globale ; Summary Plot
    st.markdown("#### *Interprétation globale des variables impactant l'octroi-crédit*")
    slice_choice = st.slider("Choisir un nombre de critères à visualiser…", 5, 25, 5)
    fig, ax = plt.subplots(figsize=(10, 10))
    shap.summary_plot(shap_values_train[1],
                      X_train_scaled,
                      feature_names=feature_names,
                      plot_type="bar",
                      plot_size=(4, slice_choice + 1),
                      max_display=slice_choice,)
    st.pyplot(fig)

    ## LOCAL
    with open('shap_values.pkl', 'rb') as f:
        shap_values_loaded = joblib.load(f)

    with open('feature_names.pkl', 'rb') as f:
        feature_names_loaded = joblib.load(f)

    # ohe
    scaling = clf.named_steps['preparation']

    # standard scaler
    clf_ = clf.named_steps['clf']
    X = identite_client(data, chk_id)
    X_train_scaled = scaling.transform(X)

    # Shap's Expected Value in logit format
    explainerlgbmc = shap.TreeExplainer(clf_)
    shap_values_train = explainerlgbmc.shap_values(X_train_scaled)

    if st.checkbox("Voir l'analyse locale, propre au Client ID{:.0f}".format(chk_id), value=False):
        shap.initjs()
        # Summary Plot ; Local
        st.markdown("#### <u>Variables impactant la décision d'octroi-crédit</u>", unsafe_allow_html=True)
        st.markdown("##### *Visualisation locale, propre au Client ID : {:.0f}*".format(chk_id))

        slice_choice = st.slider("Choisir un nombre de critères à visualiser…", 5, 20, 5)
        fig, ax = plt.subplots(figsize=(10, 10))
        shap.summary_plot(shap_values_train[1]-(0.206),
                          X_train_scaled,
                          feature_names=feature_names_loaded,
                          plot_type="bar",
                          plot_size=(4, slice_choice + 1),
                          max_display=slice_choice)
        st.pyplot(fig)

        # Force Plot ; Local
        st.markdown("#### <u>Forces contributives (négatives/positives) des variables</u>", unsafe_allow_html=True)
        st.markdown("##### *Visualisation locale, propre au Client ID : {:.0f}*".format(chk_id))
        st_shap(shap.force_plot(explainerlgbmc.expected_value[1]-(0.206),
                                shap_values_train[1],
                                X_train_scaled,
                                feature_names=feature_names_loaded,
                                link="logit",
                                matplotlib=False))

        # Decision Plot ; Local
        st.markdown("#### <u>Chemin de la décison d'octroi-crédit</u>", unsafe_allow_html=True)
        st.markdown("##### *Visualisation locale, propre au Client ID : {:.0f}*".format(chk_id))
        fig, ax = plt.subplots(figsize=(10, 10), dpi=200)
        shap.decision_plot(explainerlgbmc.expected_value[1],
                           shap_values_train[1],
                           X_train_scaled,
                           feature_names=feature_names,
                           link="logit",)
        st.pyplot(fig)
    else:
        st.markdown("___", unsafe_allow_html=True)

    # Description Details
    st.markdown("### <u>Détail de la signification des variables</u>", unsafe_allow_html=True)
    if st.checkbox("Voir le détail des critères de décision", value=False):
        list_features = description.index.to_list()
        feature = st.selectbox("Choisir une variable à détailler..", list_features)
        df = description.loc[description.index == feature][:1]
        st.table(df)
    else:
        st.markdown("___", unsafe_allow_html=True)


if __name__ == '__main__':
    main()
