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

st.set_page_config(page_title="LateGradPrecit", layout="centered")

st.title(
    "LateGradPrecit"
)

st.subheader(
    "Prediksi Risiko Keterlambatan Kelulusan Mahasiswa"
)




st.write(
    """
    Aplikasi ini memprediksi apakah mahasiswa
    berisiko terlambat lulus atau tepat waktu.
    """
)


# ======================
# UI: INPUT MANUAL
# ======================
with st.expander("Input Manual (untuk 1 mahasiswa)", expanded=True):
    nama = st.text_input("Nama Mahasiswa (opsional)")
    nim = st.text_input("NIM (opsional)")
    gender_txt = st.selectbox("Jenis Kelamin", ["L", "P"], key="gender_manual")
    ipk = st.number_input("IPK", min_value=0.0, max_value=4.0, value=3.0, step=0.01)
    prestasi_txt = st.selectbox("Prestasi", ["Ada", "Tidak Ada"], key="prestasi_manual")
    jurnal_txt = st.selectbox("Jurnal", ["Ada", "Tidak Ada"], key="jurnal_manual")

    st.divider()
    col_a, col_b = st.columns([1, 2])
    with col_a:
        submit_manual = st.button("Prediksi", type="primary")
    with col_b:
        st.write("Ringkasan input:")
        st.dataframe(
            pd.DataFrame([
                {
                    "nama": nama if nama else "-",
                    "nim": nim if nim else "-",
                    "l/p": "L" if gender_txt == "L" else "P",
                    "ipk": ipk,
                    "prestasi": prestasi_txt,
                    "jurnal": jurnal_txt,
                }
            ]),
            use_container_width=True,
            hide_index=True,
        )

# ======================
# FUNGSI BANTU
# ======================

def _normalize_gender(x):
    if pd.isna(x):
        return None
    if isinstance(x, str):
        v = x.strip().lower()
        if v in ["l", "male", "pria", "lakilaki", "laki-laki"]:
            return 1
        if v in ["p", "female", "wanita", "perempuan"]:
            return 0
    try:
        xf = float(x)
        if xf in [0, 1]:
            return int(xf)
    except Exception:
        pass
    return None


def _normalize_binary(x, true_labels, false_labels):
    if pd.isna(x):
        return None
    if isinstance(x, str):
        v = x.strip().lower()
        if v in [t.lower() for t in true_labels]:
            return 1
        if v in [f.lower() for f in false_labels]:
            return 0
    try:
        xf = float(x)
        if xf in [0, 1]:
            return int(xf)
    except Exception:
        pass
    return None


def _preprocess_input_df(df_raw: pd.DataFrame) -> pd.DataFrame:
    required = {"l/p", "ipk", "prestasi", "jurnal"}
    missing = required - set(df_raw.columns)
    if missing:
        raise ValueError(f"Kolom wajib hilang: {sorted(missing)}")

    df = df_raw.copy()

    df["l/p"] = df["l/p"].apply(_normalize_gender)
    df["prestasi"] = df["prestasi"].apply(
        lambda v: _normalize_binary(v, true_labels=["Ada", "Yes", "Ya", "Benar", "1"], false_labels=["Tidak Ada", "Tidak", "No", "0"]) 
    )
    df["jurnal"] = df["jurnal"].apply(
        lambda v: _normalize_binary(v, true_labels=["Ada", "Yes", "Ya", "Benar", "1"], false_labels=["Tidak Ada", "Tidak", "No", "0"]) 
    )

    # ipk jadi numerik
    df["ipk"] = pd.to_numeric(df["ipk"], errors="coerce")

    # cek missing hasil normalisasi
    if df[["l/p", "ipk", "prestasi", "jurnal"]].isna().any().any():
        bad_rows = df[df[["l/p", "ipk", "prestasi", "jurnal"]].isna().any(axis=1)].index.tolist()
        raise ValueError(
            "Ada baris yang tidak bisa diproses (cek nilai l/p, ipk, prestasi, jurnal). "
            f"Baris bermasalah: {bad_rows}"
        )

    # urutkan kolom sesuai model training
    df = df[["l/p", "ipk", "prestasi", "jurnal"]]
    df = df[columns]
    return df


def _predict_df(df_features: pd.DataFrame) -> pd.DataFrame:
    pred = model.predict(df_features)
    proba = model.predict_proba(df_features)

    # diasumsikan kelas urutan model: [0,1]
    # probabilitas[0] = kelas 0 (Tepat Waktu) dan probabilitas[1] = kelas 1 (Terlambat)
    return pd.DataFrame(
        {
            "prediksi": pred,
            "prob_Tepat Waktu": proba[:, 0].astype(float),
            "prob_Terlambat": proba[:, 1].astype(float),
        }
    )


# ======================
# PREDIKSI (MANUAL)
# ======================
if submit_manual:
    gender = 1 if gender_txt == "L" else 0
    prestasi = 0 if prestasi_txt == "Tidak Ada" else 1
    jurnal = 0 if jurnal_txt == "Tidak Ada" else 1

    input_data = pd.DataFrame([
        {"l/p": gender, "ipk": ipk, "prestasi": prestasi, "jurnal": jurnal}
    ])
    input_data = input_data[columns]

    hasil = int(model.predict(input_data)[0])
    probabilitas = model.predict_proba(input_data)[0]

    if hasil == 1:
        st.error("Risiko Terlambat Lulus")
        label = "Terlambat"
    else:
        st.success("Prediksi Tepat Waktu")
        label = "Tepat Waktu"

    st.write("**Hasil & Probabilitas**")
    st.dataframe(
        pd.DataFrame(
            [
                {
                    "nama": nama if nama else "-",
                    "nim": nim if nim else "-",
                    "prediksi": label,
                    "Prob_Tepat Waktu": round(float(probabilitas[0]), 3),
                    "Prob_Terlambat": round(float(probabilitas[1]), 3),
                }
            ]
        ),
        use_container_width=True,
        hide_index=True,
    )

# ======================
# UPLOAD FILE & PREDIKSI MASSAL
# ======================
with st.expander("Upload File (Prediksi untuk banyak baris)", expanded=True):
    st.write("Upload file **CSV** atau **Excel**. Kolom yang harus ada:")
    st.caption("Contoh format (CSV) - wajib punya kolom: l/p, ipk, prestasi, jurnal. Kolom nama juga boleh.")

    st.code(
        """nama,l/p,ipk,prestasi,jurnal
Ali,L,3.40,Ada,Ada
Siti,P,2.80,Tidak Ada,Tidak Ada
""",
        language="csv",
    )

    st.caption("Catatan: Nilai 'l/p' bisa berupa **L/P** atau **1/0**. 'prestasi' & 'jurnal' bisa **Ada/Tidak Ada** atau **1/0**.")

    uploaded = st.file_uploader("Pilih file", type=["csv", "xlsx", "xls"], accept_multiple_files=False)

    if uploaded is not None:
        try:
            filename = uploaded.name
            st.info(f"File terdeteksi: {filename}")

            if filename.lower().endswith(".csv"):
                df_raw = pd.read_csv(uploaded)
            else:
                df_raw = pd.read_excel(uploaded)

            st.write("**Preview data yang diupload**")
            st.dataframe(df_raw.head(50), use_container_width=True)

            st.divider()
            if st.button("Prediksi dari File", type="primary", key="predict_file"):
                df_features = _preprocess_input_df(df_raw)
                df_pred = _predict_df(df_features)

                df_result = df_raw.copy()
                df_result["prediksi"] = df_pred["prediksi"].map({1: "Terlambat", 0: "Tepat Waktu"})
                df_result["prob_Tepat Waktu"] = df_pred["prob_Tepat Waktu"].round(3)
                df_result["prob_Terlambat"] = df_pred["prob_Terlambat"].round(3)

                st.write("## Hasil Prediksi")
                st.dataframe(df_result, use_container_width=True)

                csv_out = df_result.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "Download hasil (CSV)",
                    data=csv_out,
                    file_name="hasil_prediksi.csv",
                    mime="text/csv",
                    key="download_csv",
                )

        except Exception as e:
            st.error(f"Gagal memproses file: {e}")

