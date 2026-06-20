import streamlit as st
import pandas as pd
import joblib


# ======================
# LOAD MODEL
# ======================

model = joblib.load(
    "random_forest_model.pkl"
)

columns = joblib.load(
    "columns.pkl"
)


# ======================
# JUDUL
# ======================

st.title(
    "Prediksi Risiko Keterlambatan Kelulusan Mahasiswa"
)


st.write(
    """
    Aplikasi ini memprediksi apakah mahasiswa
    berisiko terlambat lulus atau tepat waktu.
    """
)


# ======================
# INPUT
# ======================

gender = st.selectbox(
    "Jenis Kelamin",
    ["L", "P"]
)


ipk = st.number_input(
    "IPK",
    min_value=0.0,
    max_value=4.0,
    value=3.0
)


prestasi = st.selectbox(
    "Prestasi",
    ["Ada", "Tidak Ada"]
)


jurnal = st.selectbox(
    "Jurnal",
    ["Ada", "Tidak Ada"]
)



# ======================
# PREDIKSI
# ======================

if st.button("Prediksi"):


    # preprocessing sama seperti training

    gender = 1 if gender == "L" else 0

    prestasi = 0 if prestasi == "Tidak Ada" else 1

    jurnal = 0 if jurnal == "Tidak Ada" else 1


    input_data = pd.DataFrame([{

        "l/p": gender,
        "ipk": ipk,
        "prestasi": prestasi,
        "jurnal": jurnal

    }])


    # memastikan urutan kolom sama

    input_data = input_data[columns]


    hasil = model.predict(
        input_data
    )[0]


    probabilitas = model.predict_proba(
        input_data
    )[0]


    if hasil == 1:

        st.error(
            "Risiko Terlambat Lulus"
        )

    else:

        st.success(
            "Prediksi Tepat Waktu"
        )


    st.write(
        "Probabilitas:"
    )


    st.write(
        {
            "Tepat Waktu": round(float(probabilitas[0]),3),
            "Terlambat": round(float(probabilitas[1]),3)
        }
    )