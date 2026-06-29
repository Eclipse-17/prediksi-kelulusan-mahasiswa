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

st.title("LateGradPrecit")

st.subheader("Prediksi Risiko Keterlambatan Kelulusan Mahasiswa")

st.markdown(
    """
    Aplikasi ini memprediksi apakah mahasiswa **berisiko terlambat lulus** atau **tepat waktu**.

    Gunakan tab di atas untuk:
    - **Input Manual** (1 mahasiswa)
    - **Upload File** (banyak baris)
    """
)



# ======================
# UI: HALAMAN (TAB)
# ======================
page_tabs = st.tabs(["Input Manual", "Upload File", "Dataset & Train"])






# ====== Tab 1: INPUT MANUAL ======
with page_tabs[0]:
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
            submit_manual = st.button("✨ Prediksi", type="primary", key="submit_manual")
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
                width="stretch",
                hide_index=True,
            )


# ====== Tab 2: UPLOAD FILE ======
# (UI form ada di bagian prediksi massal di bawah)
with page_tabs[1]:
    st.write("")



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
# PREDIKSI (MANUAL) -> Tab 1
# ======================
with page_tabs[0]:
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
        # Badge warna untuk hasil
        is_terlambat = (hasil == 1)
        badge_html = (
            "<div style='padding:10px 14px; border-radius:12px; font-weight:700; "
            f"background-color:{'#FEE2E2' if is_terlambat else '#DCFCE7'}; color:{'#991B1B' if is_terlambat else '#065F46'}; "
            "text-align:center; border:1px solid; border-color:inherit;'>"
            f"{label}"
            "</div>"
        )
        st.markdown(badge_html, unsafe_allow_html=True)

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
            width="stretch",
            hide_index=True,
        )



# ======================
# UPLOAD FILE & PREDIKSI MASSAL -> Tab 2
# ======================
with page_tabs[1]:

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


# ======================
# DATASET & TRAIN -> Tab 3
# ======================
with page_tabs[2]:
    st.subheader("Dataset & Train RandomForest")

    # Simpan history training di session state
    if "model_history" not in st.session_state:
        st.session_state.model_history = []

    # (Opsional) area riwayat ringkas di tab dataset
    if st.session_state.model_history:
        st.info(f"Riwayat training tersimpan di session: {len(st.session_state.model_history)} model")


    # =========================================================================

    # 1. PANDUAN FORMAT (Selalu Muncul di Paling Atas Mengikuti Contoh Anda)
    # =========================================================================
    st.markdown("### 📋 Panduan Format Dataset")
    st.markdown(
        "Sebelum mengunggah, pastikan file CSV/Excel Anda memiliki struktur kolom "
        "dan format data seperti contoh di bawah ini:"
    )
    
    # Membuat data dummy persis sesuai contoh yang Anda berikan
    df_contoh = pd.DataFrame([
        {"nama": "Ali", "l/p": "L", "ipk": 3.40, "prestasi": "Ada", "jurnal": "Ada", "tepat waktu": "terlambat"},
        {"nama": "Siti", "l/p": "P", "ipk": 2.80, "prestasi": "Tidak Ada", "jurnal": "Tidak Ada", "tepat waktu": "tepat waktu"},
        {"nama": "Budi", "l/p": "L", "ipk": 3.10, "prestasi": "Ada", "jurnal": "Tidak Ada", "tepat waktu": "tepat waktu"},
        {"nama": "Rina", "l/p": "P", "ipk": 2.50, "prestasi": "Tidak Ada", "jurnal": "Ada", "tepat waktu": "terlambat"},
        {"nama": "Doni", "l/p": "L", "ipk": 3.60, "prestasi": "Ada", "jurnal": "Ada", "tepat waktu": "tepat waktu"}
    ])
    
    st.dataframe(df_contoh, use_container_width=True, hide_index=True)
    st.caption("💡 *Pastikan file Anda memiliki kolom target dengan nama `tepat waktu` (atau variasi serupa).*")
    st.divider()

    # =========================================================================
    # 2. COMPONENT UPLOAD FILE
    # =========================================================================
    uploaded_train = st.file_uploader(
        "Upload dataset (CSV/XLSX/XLS)",
        type=["csv", "xlsx", "xls"],
        accept_multiple_files=False,
        key="upload_train_tab3",
    )

    if uploaded_train is not None:
        try:
            filename = uploaded_train.name
            st.info(f"Dataset terdeteksi: {filename}")

            # Membaca file berdasarkan format extension
            if filename.lower().endswith(".csv"):
                df_raw = pd.read_csv(uploaded_train)
            else:
                df_raw = pd.read_excel(uploaded_train)

            st.write("Preview dataset Anda (10 baris):")
            st.dataframe(df_raw.head(10), use_container_width=True)

            st.divider()

            # =========================================================================
            # 3. PROSES VALIDASI & PENCARIAN KOLOM LABEL
            # =========================================================================
            label_col = None
            # Urutan pencarian kata kunci kolom label
            for cand in ["tepat waktu", "tepat_waktu", "terlambat", "status_kelulusan", "status", "label"]:
                for c in df_raw.columns:
                    if str(c).strip().lower() == cand:
                        label_col = c
                        break
                if label_col is not None:
                    break

            # Pencarian substring jika belum ketemu yang persis
            if label_col is None:
                for c in df_raw.columns:
                    lc = str(c).strip().lower()
                    if "tepat" in lc or "terlambat" in lc or "status" in lc or "label" in lc:
                        label_col = c
                        break

            # Jika kolom label tidak ditemukan, hentikan proses dan beri tahu pengguna
            if label_col is None:
                raise ValueError(
                    "Tidak menemukan kolom label/target di file Anda. "
                    "Pastikan file yang diupload memiliki kolom target bernama 'tepat waktu' seperti panduan di atas."
                )

            st.write(f"Kolom label terdeteksi: `{label_col}`")

            # =========================================================================
            # 4. MAPPING DATA & PREPROCESSING
            # =========================================================================
            def _map_label(v):
                if pd.isna(v):
                    return None
                s = str(v).strip().lower()
                if s in ["terlambat", "telat", "1", "ya", "yes", "y", "true"]:
                    return 1
                if s in ["tepat waktu", "tepat", "0", "tidak", "no", "false"]:
                    return 0
                if "terlambat" in s or "telat" in s:
                    return 1
                if "tepat" in s and "waktu" in s:
                    return 0
                return None

            df_tmp = df_raw.copy()
            df_tmp["__y"] = df_tmp[label_col].apply(_map_label)

            if df_tmp["__y"].isna().any():
                bad_idx = df_tmp[df_tmp["__y"].isna()].index.tolist()
                raise ValueError(f"Ada nilai label kelulusan yang tidak dikenali pada baris: {bad_idx[:20]}{'...' if len(bad_idx)>20 else ''}")

            # Validasi keberadaan kolom fitur wajib
            required_feat = ["l/p", "ipk", "prestasi", "jurnal"]
            missing_feat = [c for c in required_feat if c not in df_tmp.columns]
            if missing_feat:
                raise ValueError(f"Kolom fitur wajib hilang dari file Anda: {missing_feat}")

            # Pemrosesan dataframe fitur lewat fungsi eksternal Anda
            df_X_raw = df_tmp[required_feat].copy()
            df_X = _preprocess_input_df(df_X_raw)
            y = df_tmp["__y"].astype(int).values

            # =========================================================================
            # 5. MODEL TRAINING (RANDOM FOREST)
            # =========================================================================
            from sklearn.model_selection import train_test_split
            from sklearn.ensemble import RandomForestClassifier
            from sklearn.metrics import accuracy_score

            X_train, X_test, y_train, y_test = train_test_split(
                df_X, y, test_size=0.2, random_state=42, stratify=y
            )

            clf = RandomForestClassifier(n_estimators=300, random_state=42)
            clf.fit(X_train, y_train)

            y_pred = clf.predict(X_test)
            acc = accuracy_score(y_test, y_pred)
            st.success(f"Model Berhasil Dilatih! Akurasi Uji (Test Accuracy) = {acc:.4f}")

            # Simpan ke riwayat model (history)
            # (pilihan: hanya session ini, tidak persist ke file)
            try:
                import datetime
                model_item = {
                    "trained_at": datetime.datetime.now().isoformat(timespec="seconds"),
                    "dataset_name": getattr(uploaded_train, "name", "dataset"),
                    "accuracy": float(acc),
                    "n_rows": int(df_raw.shape[0]),
                    "n_cols": int(df_raw.shape[1]),
                    "label_col": str(label_col),
                    "clf": clf,
                    "columns": list(columns),
                    "features": required_feat,
                }
                st.session_state.model_history.append(model_item)
            except Exception:
                # jika gagal simpan, tetap lanjut UI training
                pass

            st.divider()

            
            # =========================================================================
            # 6. FORM PREDIKSI MANUAL (MENGGUNAKAN MODEL BARU TAB 3)
            # =========================================================================
            st.subheader("Prediksi Manual (menggunakan model hasil training Tab 3)")

            with st.expander("Isi data 1 mahasiswa", expanded=True):
                nama3 = st.text_input("Nama Mahasiswa (opsional)", key="nama_tab3")
                nim3 = st.text_input("NIM (opsional)", key="nim_tab3")
                gender_txt3 = st.selectbox("Jenis Kelamin", ["L", "P"], key="gender_tab3")
                ipk3 = st.number_input("IPK", min_value=0.0, max_value=4.0, value=3.0, step=0.01, key="ipk_tab3")
                prestasi_txt3 = st.selectbox("Prestasi", ["Ada", "Tidak Ada"], key="prestasi_tab3")
                jurnal_txt3 = st.selectbox("Jurnal", ["Ada", "Tidak Ada"], key="jurnal_tab3")

                if st.button("Prediksi", type="primary", key="pred_tab3_btn"):
                    df_one_raw = pd.DataFrame([
                        {
                            "l/p": "L" if gender_txt3 == "L" else "P",
                            "ipk": ipk3,
                            "prestasi": prestasi_txt3,
                            "jurnal": jurnal_txt3,
                        }
                    ])

                    df_one_X = _preprocess_input_df(df_one_raw)
                    y_one = int(clf.predict(df_one_X)[0])
                    proba_one = clf.predict_proba(df_one_X)[0]

                    if y_one == 1:
                        label = "Terlambat"
                        st.error("Risiko Terlambat Lulus")
                    else:
                        label = "Tepat Waktu"
                        st.success("Prediksi Tepat Waktu")

                    st.write("**Hasil & Probabilitas**")
                    st.dataframe(
                        pd.DataFrame([
                            {
                                "nama": nama3 if nama3 else "-",
                                "nim": nim3 if nim3 else "-",
                                "prediksi": label,
                                "Prob_Tepat Waktu": round(float(proba_one[0]), 3),
                                "Prob_Terlambat": round(float(proba_one[1]), 3),
                            }
                        ]),
                        use_container_width=True,
                        hide_index=True,
                    )

        except Exception as e:
            st.error(f"Gagal memproses dataset training: {e}")
