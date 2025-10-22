import os
import streamlit as st
import pandas as pd
import numpy as np
import joblib
from io import BytesIO

# ========= إعداد الصفحة =========
st.set_page_config(page_title="نظام تحليل الفقد الذكي", page_icon="⚡", layout="wide")
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

# ========= مسارات =========
if os.name == "nt":
    model_folder = r"C:\asd6"
else:
    model_folder = "asd6"
os.makedirs(model_folder, exist_ok=True)

model_path = os.path.join(model_folder, "ASD6.pkl")       # ⚠️ نموذج جاهز وموجود مسبقًا
data_frame_template_path = "The data frame file to be analyzed.xlsx"

# ========= كاش تحميل النموذج (قراءة فقط) =========
@st.cache_resource
def load_model_once():
    if not os.path.exists(model_path):
        st.error("⚠️ ملف النموذج غير موجود: ASD6.pkl — ارفعه في نفس مجلد التطبيق.")
        st.stop()
    return joblib.load(model_path)

# ========= كاش قراءة ملفات الإدخال =========
@st.cache_data
def read_excel_cached(upload):
    # تقليل الذاكرة: read only needed cols إن حبيت
    df = pd.read_excel(upload)
    # حوّل الأعمدة الرقمية إلى float32 لتقليل RAM
    for c in ["V1","V2","V3","A1","A2","A3"]:
        if c in df.columns:
            df[c] = df[c].astype("float32")
    return df

model = load_model_once()  # يُحمَّل مرة واحدة فقط

# ========= بقية إعداداتك/سلايدراتك كالمعتاد =========
# ... I_abs_min, I_rel_min, ... إلخ

# ========= التحليل =========
def analyze_data(df):
    required = {'V1','V2','V3','A1','A2','A3','Meter Number'}
    missing = required - set(df.columns)
    if missing:
        st.error("⚠️ الملف ينقصه الأعمدة التالية: " + ", ".join(missing))
        return

    X = df[['V1','V2','V3','A1','A2','A3']]  # صارت float32
    df = df.copy()
    # تنبؤ (لا تدريب مطلقًا)
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(X)[:,1]
        df["Model_Prob"] = probs.astype("float32")
    df["Predicted_Loss"] = model.predict(X)

    # ← ضع هنا دوالك الهندسية: detect_two_phase_strict / reason_engine / severity_score ...
    # df["Reason"] = ...
    # df["Severity_Score"] = ...
    # df["Priority"] = ...

    # أمثلة لتخفيف التصدير: CSV افتراضيًا
    st.download_button(
        "📥 تنزيل النتائج (CSV موفّر للذاكرة)",
        data=df.to_csv(index=False).encode("utf-8"),
        file_name="loss_analysis.csv",
        mime="text/csv"
    )

    # لو تبغى Excel لملخصات صغيرة فقط:
    # small = df.head(30000)  # مثال حد أعلى
    # out = BytesIO(); small.to_excel(out, index=False); out.seek(0)
    # st.download_button("تنزيل Excel (عيّنة مختصرة)", data=out, file_name="loss_analysis_sample.xlsx")

    # عرض الجداول كما لديك
    # st.dataframe(...)

# ========= واجهة الرفع =========
st.header("📤 تحليل ملف Excel")
upload = st.file_uploader("ارفع الملف", type=["xlsx"])
if upload is not None:
    df = read_excel_cached(upload)
    analyze_data(df)
