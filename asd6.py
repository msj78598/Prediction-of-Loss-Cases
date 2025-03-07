import os
import streamlit as st
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from io import BytesIO

# حل مشكلة protobuf في بيئة Streamlit Cloud
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

# ضبط المسار بناءً على نظام التشغيل
if os.name == "nt":  # Windows
    model_folder = "C:\\asd6"
else:  # Linux (مثل Streamlit Cloud)
    model_folder = "asd6"

if not os.path.exists(model_folder):
    os.makedirs(model_folder)

model_path = os.path.join(model_folder, 'ASD6.pkl')
data_frame_template_path = 'The data frame file to be analyzed.xlsx'

# تحميل البيانات وتدريب النموذج إذا لم يكن موجودًا
def train_and_save_model():
    try:
        file_path = 'final_classified_loss_with_reasons_60_percent_ordered.xlsx'
        
        if not os.path.exists(file_path):
            st.error(f"⚠️ ملف التدريب {file_path} غير موجود! يرجى رفعه إلى المجلد الرئيسي.")
            return
        
        data = pd.read_excel(file_path)
        X = data[['V1', 'V2', 'V3', 'A1', 'A2', 'A3']]
        y = data['Loss_Status'].apply(lambda x: 1 if x == 'Loss' else 0)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        model = RandomForestClassifier(random_state=42)
        model.fit(X_train, y_train)

        joblib.dump(model, model_path)
        st.success(f"✅ تم تدريب النموذج وحفظه في {model_path}")
    
    except Exception as e:
        st.error(f"❌ حدث خطأ أثناء تدريب النموذج: {str(e)}")

if not os.path.exists(model_path):
    train_and_save_model()

# دالة تحليل البيانات
def analyze_data(data):
    try:
        if 'Meter Number' not in data.columns:
            st.error("⚠️ الملف لا يحتوي على عمود 'Meter Number'. يرجى التحقق من البيانات.")
            return
        
        model = joblib.load(model_path)
        X = data[['V1', 'V2', 'V3', 'A1', 'A2', 'A3']]
        predictions = model.predict(X)

        data['Predicted_Loss'] = predictions
        loss_data = data[data['Predicted_Loss'] == 1].copy()
        
        st.write(f"🔍 عدد حالات الفقد المكتشفة: **{len(loss_data)}**")

        st.subheader("📋 جميع حالات الفقد المكتشفة")
        st.dataframe(loss_data)

        output_loss = BytesIO()
        with pd.ExcelWriter(output_loss, engine='xlsxwriter') as writer:
            loss_data.to_excel(writer, index=False)
        output_loss.seek(0)

        st.download_button("📥 تحميل جميع حالات الفقد", data=output_loss, file_name="all_loss_cases.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

    except Exception as e:
        st.error(f"❌ حدث خطأ أثناء تحليل البيانات: {str(e)}")

# Streamlit App
st.title("🔌 التنبؤ بحالات الفقد المحتملة")

# تحميل نموذج البيانات
if os.path.exists(data_frame_template_path):
    with open(data_frame_template_path, 'rb') as template_file:
        template_data = template_file.read()
    st.download_button("📥 تحميل قالب البيانات", data=template_data, file_name="The_data_frame_file_to_be_analyzed.xlsx")
else:
    st.warning("⚠️ قالب البيانات غير متوفر! يرجى رفعه إلى المجلد الرئيسي.")

st.header("📊 تحليل البيانات")
uploaded_analyze_file = st.file_uploader("📤 قم برفع ملف البيانات للتحليل (Excel)", type=["xlsx"])
if uploaded_analyze_file is not None:
    try:
        analyze_data_df = pd.read_excel(uploaded_analyze_file)
        analyze_data(analyze_data_df)
    except Exception as e:
        st.error(f"❌ خطأ في تحميل الملف: {str(e)}")

st.markdown("---")
st.title("👨‍💻 المطور: **مشهور العباس**")
