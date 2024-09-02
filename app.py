import streamlit as st
import numpy as np
import joblib

# Charger le modèle dans Streamlit
with open('model_logreg.pkl', 'rb') as file:
    model = joblib.load(file)


# Titre de l'application
st.title("Prédiction de la Qualité du Sommeil")

# Entrée des SAS (1 à 6 pour chaque question)
sas1 = st.slider("SAS1: J'utilise mon smartphone de telle manière à ce que cela entraîne un impact négatif sur ma productivité/mon travail.", 1, 6)
sas2 = st.slider("SAS2: J'ai du mal à me concentrer en classe, durant mes devoirs, ou durant le travail à cause du Smartphone.", 1, 6)
sas3 = st.slider("SAS3: Je ressens de la douleur aux poignets où à la nuque quand j'utilise mon Smartphone.", 1, 6)
sas4 = st.slider("SAS4: Je ne supporte pas le fait de ne pas avoir mon Smartphone", 1, 6)
sas5 = st.slider("SAS5: Je ressens de l'impatience et de l'irritation lorsque je n'ai pas mon Smartphone", 1, 6)
sas6 = st.slider("SAS6: Je suis préoccupé par l'utilisation de mon Smartphone, même lorsque je ne l'utilise pas.", 1, 6)
sas7 = st.slider("SAS7: Je n'arrêterai jamais d'utiliser mon Smartphone même si son utilisation entraîne des conséquences négatives importantes dans ma vie quotidienne.", 1, 6)
sas8 = st.slider("SAS8: Je surveille en permanence mon Smartphone de manière à ne manquer aucune conversation (par exemple : twitter, Facebook).", 1, 6)
sas9 = st.slider("SAS9: J’utilise mon smartphone plus longtemps que je ne l’avais prévu", 1, 6)
sas10 = st.slider("SAS10: Mes proches me disent que j'utilise trop mon Smartphone", 1, 6)

# Calculer le score d'addiction
score_addiction = sum([sas1, sas2, sas3, sas4, sas5, sas6, sas7, sas8, sas9, sas10])

st.write(f"Score d'addiction calculé : {score_addiction}")

# Entrée des autres variables indépendantes
u3 = st.selectbox("Accès à internet sur le smartphone", ["Jamais", "Parfois", "Souvent", "Toujours"])
u3 = {"Jamais": 1, "Parfois": 2, "Souvent": 3, "Toujours": 4}[u3]


u4 = st.selectbox("Utilisation du smartphone les jours d'école", ["Moins d'une heure par jour", "Entre 1 et 4 heures par jour", "Plus de 4 heures"])
u4 = {"Moins d'une heure par jour": 1, "Entre 1 et 4 heures par jour": 2, "Plus de 4 heures": 3}[u4]

u5 = st.selectbox("Utilisation du smartphone pendant les vacances", ["Moins d'une heure", "Entre 1 et 4 heures par jour", "Plus de 4 heures"])
u5 = {"Moins d'une heure": 1, "Entre 1 et 4 heures par jour": 2, "Plus de 4 heures": 3}[u5]


u7 = st.selectbox("Consultation du smartphone au réveil", ["Dès que j’ouvre les yeux", "Dans les 15 minutes après mon réveil", "15 à 30 min après mon réveil", "Plus de 30 min après mon réveil"])
u7 = {"Dès que j’ouvre les yeux": 1, "Dans les 15 minutes après mon réveil": 2, "15 à 30 min après mon réveil": 3, "Plus de 30 min après mon réveil": 4}[u7]


u8 = st.selectbox("Pose du smartphone le soir", ["Je m’endors en l’utilisant", "Moins d’une heure avant d’aller me coucher", "Plus d’une heure avant de me coucher"])
u8 = {"Je m’endors en l’utilisant": 1, "Moins d’une heure avant d’aller me coucher": 2, "Plus d’une heure avant de me coucher": 3}[u8]


u11 = st.selectbox(" Il m’arrive d’avoir des difficultés à me concentrer parce que j’utilise mon smartphone", ["OUI", "Non"])
u11 = {"OUI": 1, "Non": 2}[u11]

u12 = st.selectbox(" Il m’arrive d’avoir des difficultés à me concentrer parce que je pense à mon smartphone :", ["OUI", "Non"])
u12 = {"OUI": 1, "Non": 2}[u12]

u13 = st.selectbox("J’oublie des choses importantes à cause de mon smartphone", ["OUI", "Non"])
u13 = {"OUI": 1, "Non": 2}[u13]

u14 = st.selectbox("Je préfère utiliser mon smartphone que participer à des activités avec d’autres personnes", ["Jamais", "Parfois", "Souvent", "Toujours"])
u14 = {"Jamais": 1, "Parfois": 2, "Souvent": 3, "Toujours": 4}[u14]

u16 = st.selectbox(". Il m’arrive de préférer utiliser mon smartphone que de faire un travail ou un devoir", ["OUI", "Non"])
u16 = {"OUI": 1, "Non": 2}[u16]


# Rassembler les variables dans un tableau
features = np.array([[score_addiction, u3, u4, u5, u7, u8, u11, u12, u13, u14, u16]])

# Prédiction
if st.button("Prédire"):
    prediction = model.predict(features)
    prediction_proba = model.predict_proba(features)[0][1]
    
    st.success(f"La qualité de sommeil prévue est : {prediction[0]}")
    st.info(f"Probabilité associée à cette prédiction : {prediction_proba:.2f}")

    # Suggestions basées sur la prédiction
    if prediction[0] == 0:  # 1 pour mauvaise qualité de sommeil
        st.warning("Votre qualité de sommeil semble mauvaise. Essayez de réduire l'utilisation de votre smartphone avant de dormir.")
    else:
        st.info("Votre qualité de sommeil semble bonne. Continuez à maintenir un bon équilibre.")
