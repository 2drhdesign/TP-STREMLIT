import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris


st.title("Bienvenue sur l'application Iris")
st.subheader("Développée par Donald HOUNGBEDJI")


def load_data():
    iris = load_iris()
    data = pd.DataFrame(data=iris.data, columns=iris.feature_names)
    data['species'] = iris.target
    data['species'] = data['species'].apply(lambda x: iris.target_names[x])
    return data


st.title("Données Iris à analyser  ")


iris_data = load_data()


st.header("Données Iris")
if st.checkbox("Afficher les données brutes"):
    st.dataframe(iris_data)


st.header("Graphiques dynamiques 📊")
columns = iris_data.columns[:-1]


x_axis = st.selectbox("Choisissez la colonne pour l'axe X:", columns)
y_axis = st.selectbox("Choisissez la colonne pour l'axe Y:", columns)
hue = st.selectbox("Choisissez une colonne pour le regroupement (couleur):", ['species'])


st.subheader(f"Graphique : {x_axis} vs {y_axis}")
fig, ax = plt.subplots()
sns.scatterplot(data=iris_data, x=x_axis, y=y_axis, hue=hue, ax=ax)
st.pyplot(fig)


st.subheader("Distribution des colonnes")
col_to_plot = st.selectbox("Choisissez une colonne pour la distribution:", columns)
fig_dist, ax_dist = plt.subplots()
sns.histplot(iris_data[col_to_plot], kde=True, ax=ax_dist)
st.pyplot(fig_dist)


st.header("Statistiques descriptives 📈")
st.write(iris_data.describe())
