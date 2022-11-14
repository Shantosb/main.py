import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns



@st.cache
def load_data():
    df = pd.read_csv("breast-cancer.csv")
    df = df.drop(["id", "fractal_dimension_mean", "texture_se", "smoothness_se", "symmetry_se", "fractal_dimension_se"],
                 axis=1)
    df["diagnosis"] = df["diagnosis"].replace(["B", "M"], ["0", "1"])
    df['diagnosis'] = df['diagnosis'].astype('int')
    return df


df = load_data()


def show_explore_page():
    st.title("Data information")

    st.write("How the data relate to the diagnosis")
    st.write("Heatmap showing the levels of correlation between each attribute with 1 being the highest possible number")
    df.corr()
    corr_matrix = df.corr()
    fig, ax = plt.subplots(figsize=(20,20))
    ax = sns.heatmap(corr_matrix,
                     annot=True,
                     linewidths=0.5,
                     fmt=".2f",
                     cmap="YlGnBu");
    st.pyplot(fig)

    x_benign = df[df['diagnosis'] == 0]
    x_benign = x_benign.drop(["diagnosis"], axis=1)
    x_benign.columns = ['R.M.', 'T.M.', 'P.M.', 'A.M.', 'S.M.', 'C.M.', 'CON.M.', 'CON.P.M', 'SYM.M', 'R.S.', 'P.S',
                        'A.S', 'C.S.', 'CON.S.',
                        'CON.P.S', 'R.W', 'T.W.', 'P.W', 'A.W', 'S.W.', 'COM.W', 'CON.W', 'CON.P.W', 'S.W', 'F.D.W.']
    x_benign_avg = x_benign.mean()
    x_benign_avg = x_benign_avg.round(3)


    x_malign = df[df['diagnosis'] == 1]
    x_malign = x_malign.drop(["diagnosis"], axis=1)
    x_malign.columns = ['R.M.', 'T.M.', 'P.M.', 'A.M.', 'S.M.', 'C.M.', 'CON.M.', 'CON.P.M', 'SYM.M', 'R.S.', 'P.S',
                        'A.S', 'C.S.', 'CON.S.',
                        'CON.P.S', 'R.W', 'T.W.', 'P.W', 'A.W', 'S.W.', 'COM.W', 'CON.W', 'CON.P.W', 'S.W', 'F.D.W.']
    x_malign_avg = x_malign.mean()
    x_malign_avg = x_malign_avg.round(3)

    st.write("##")
    st.write("##")
    st.write("BarChart showing the mean value of every characteristic used to diagnose a tumor as Benign")

    fig2, (ax2) = plt.subplots(1, figsize=(30, 10))

    p1 = ax2.bar(x_benign_avg.keys(), x_benign_avg, color=(0.572549, 0.2862, 0.0, 1))

    ax2.set_xlabel('Parameters', fontsize=40)
    ax2.set_ylabel('AVG. Values', fontsize=40)
    ax2.set_yscale('log')
    ax2.set_title('Benign Tumor AVG. Chars.', fontsize =40)
    ax2.bar_label(container=p1, label=x_benign_avg, fontsize=20)

    fig2.tight_layout()
    st.pyplot(fig2)

    st.write("#")
    st.write("BarChart showing the mean value of every characteristic used to diagnose a tumor as Malignant")
    fig3, (ax3) = plt.subplots(1, figsize=(30, 10))

    p2 = ax3.bar(x_malign_avg.keys(), x_malign_avg)

    ax3.set_xlabel('Parameters', fontsize=40)
    ax3.set_ylabel('AVG. Values', fontsize=40)
    ax3.set_yscale('log')
    ax3.set_title('Malignant Tumor AVG. Chars.', fontsize=40)
    ax3.bar_label(container=p2, label=x_malign_avg, fontsize=20)

    fig3.tight_layout()
    st.pyplot(fig3)