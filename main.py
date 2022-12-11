import pandas as pd
import streamlit as st
import numpy as np
import altair as alt
from sklearn.preprocessing import MinMaxScaler
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
 

st.write(""" 
# Kumpulan Data Kualitas Anggur
by MT ALDI SYAHPRANATA (200411100175)
""")

st.write("----------------------------------------------------------------------------------")

des, dataset, proses, model, implementasi = st.tabs(["Deskripsi", "Dataset", "Preprocessing", "Model", "Implementasi"])

with des:
    st.subheader("""Pengertian""")
    st.write(""" 
    Dataset ini terkait dengan varian merah anggur "Vinho Verde" Portugis. Kumpulan data menggambarkan jumlah berbagai bahan kimia yang ada dalam anggur dan pengaruhnya terhadap kualitasnya. Kumpulan data dapat dilihat sebagai tugas klasifikasi atau regresi. Kelas dipesan dan tidak seimbang (mis. ada lebih banyak anggur normal daripada anggur yang sangat baik atau buruk).
    """)

    st.markdown(
        """
        fitur 
        
        1-keasaman tetap
        
        2-keasaman volatil

        3-asam sitrat

        4-sisa gula

        5-klorida

        6 bebas sulfur dioksida)

        7-total sulfur dioksida)

        8-kepadatan

        9-pH

        10-sulfat

        11-alkohol

        Variabel keluaran (berdasarkan data sensorik); 
        
        12 kualitas (skor antara 0 dan 10)
        """
    )

with dataset:
    st.subheader("""Dataset""")
    df = pd.read_csv('https://raw.githubusercontent.com/Aldi-880/datamining/main/WineQT.csv')
    st.dataframe(df) 

with proses:
    st.subheader("""Rumus Normalisasi Data""")
    st.image('https://i.stack.imgur.com/EuitP.png', use_column_width=False, width=250)
    st.markdown("""
    Keterangan :
    - X = data yang akan dinormalisasi atau data asli
    - min = nilai minimum semua data asli
    - max = nilai maksimum semua data asli
    """)
    #Mendefinisikan Varible X dan Y
    X = df.drop(columns=['alcohol', 'quality', 'Id'])
    y = df['quality'].values
    df_min = X.min()
    df_max = X.max()
    
    #NORMALISASI NILAI X
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(X)
    features_names = X.columns.copy()
    scaled_features = pd.DataFrame(scaled, columns=features_names)

    st.subheader('Hasil Normalisasi Data')
    st.write(scaled_features)

    st.subheader('Target Label')
    dumies = pd.get_dummies(df.quality).columns.values.tolist()
    dumies = np.array(dumies)

    labels = pd.DataFrame({
        'Positive' : [dumies[1]],
        'Negative' : [dumies[0]]
    })

    st.write(labels)

with model:
    #Split Data 
    training, test = train_test_split(scaled_features,test_size=0.2, random_state=1)#Nilai X training dan Nilai X testing
    training_label, test_label = train_test_split(y, test_size=0.2, random_state=1)#Nilai Y training dan Nilai Y testing
    with st.form("modeling"):
        st.write("Pilih Metode : ")
        naive = st.checkbox('Gaussian Naive Bayes')
        destree = st.checkbox('Decission Tree')
        k_nn = st.checkbox('K-NN')
        submitted = st.form_submit_button("Submit")

        #Gaussian Naive Bayes
        gaussian = GaussianNB()
        gaussian = gaussian.fit(training, training_label)

        probas = gaussian.predict_proba(test)
        probas = probas[:,1]
        probas = probas.round()

        gaussian_akurasi = round(100 * accuracy_score(test_label,probas))

        #KNN
        K=10
        knn=KNeighborsClassifier(n_neighbors=K)
        knn.fit(training,training_label)
        knn_predict=knn.predict(test)

        knn_akurasi = round(100 * accuracy_score(test_label,knn_predict))

        #Decission Tree
        dt = DecisionTreeClassifier()
        dt.fit(training, training_label)
        # prediction
        dt_pred = dt.predict(test)
        #Accuracy
        dt_akurasi = round(100 * accuracy_score(test_label,dt_pred))

        if submitted :
            if naive :
                st.write('Naive Bayes accuracy score: {0:0.2f}'. format(gaussian_akurasi))
            if destree :
                st.write("Decision Tree accuracy score : {0:0.2f}" . format(dt_akurasi))
            if k_nn :
                st.write("K-NN accuracy score : {0:0.2f}" . format(knn_akurasi))
            
        
        grafik = st.form_submit_button("Grafik Akurasi")
        if grafik:
            data = pd.DataFrame({
                'Akurasi' : [gaussian_akurasi, knn_akurasi, dt_akurasi],
                'Model' : ['Gaussian Naive Bayes', 'K-NN', 'Decission Tree'],
            })

            chart = (
                alt.Chart(data)
                .mark_bar()
                .encode(
                    alt.X("Akurasi"),
                    alt.Y("Model"),
                    alt.Color("Akurasi"),
                    alt.Tooltip(["Akurasi", "Model"]),
                )
                .interactive()
            )
            st.altair_chart(chart,use_container_width=True)

with implementasi:
        with st.form("my_form"):
            fixed = st.slider('Nilai keasaman tetap', 4.6, 16.9)
            volatile = st.slider('Nilai keasaman yang mudah menguap', 0.12, 1.58)
            citric = st.slider('Nilai asam sitrat', 0, 1)
            residual = st.slider('Nilai sisa gula', 0.9, 15.5)
            chlorides = st.slider('Nilai klorida', 0.01, 0.61)
            freesulfur = st.slider('Nilai sulfur dioksida gratis', 1, 68)
            totalsulfur = st.slider('Nilai sulfur dioksida total', 6, 289)
            density = st.slider('Nilai kepadatan', 0.99, 1.00)
            ph = st.slider('nilai pH', 2.74, 4.01)
            sulphates = st.slider('Nilai sulfat', 0.33, 2.00)
            model = st.selectbox('Model untuk prediksi',
                    ('Gaussian Naive Bayes', 'K-NN', 'Decision Tree'))

            prediksi = st.form_submit_button("Submit")
            if prediksi:
                inputs = np.array([
                    fixed,
                    volatile,
                    citric,
                    residual,
                    chlorides,
                    freesulfur,
                    totalsulfur,
                    density,
                    ph,
                    sulphates
                ])
                
                df_min = X.min()
                df_max = X.max()
                input_norm = ((inputs - df_min) / (df_max - df_min))
                input_norm = np.array(input_norm).reshape(1, -1)

                if model == 'Gaussian Naive Bayes':
                    mod = gaussian
                if model == 'K-NN':
                    mod = knn 
                if model == 'Decision Tree':
                    mod = dt

                input_pred = mod.predict(input_norm)

                st.subheader('Hasil Prediksi')
                st.write('Menggunakan Pemodelan :', model)

                if input_pred == 1:
                    st.error('Positive')
                else:
                    st.success('Negative')