import os
import streamlit as st
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from io import BytesIO

# 1) يجب أن يكون هذا أول استدعاء لأي دالة st.*
st.set_page_config(page_title="التنبؤ بحالات الفقد", page_icon="⚡", layout="wide", initial_sidebar_state="expanded")

# إعداد بيئة Streamlit وتحميل المتطلبات
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

# ضبط المسارات
if os.name == "nt":  # Windows
    model_folder = "C:\\asd6"
else:  # Linux (Streamlit Cloud)
    model_folder = "asd6"

if not os.path.exists(model_folder):
    os.makedirs(model_folder)

model_path = os.path.join(model_folder, 'ASD6.pkl')
data_frame_template_path = 'The data frame file to be analyzed.xlsx'

# --- تدريب/تحميل النموذج ---
def train_and_save_model():
    try:
        file_path = 'final_classified_loss_with_reasons_60_percent_ordered.xlsx'
        if not os.path.exists(file_path):
            st.error("⚠️ ملف التدريب غير موجود! يرجى رفعه إلى المجلد الرئيسي.")
            return False

        data = pd.read_excel(file_path)
        X = data[['V1', 'V2', 'V3', 'A1', 'A2', 'A3']]
        y = data['Loss_Status'].apply(lambda x: 1 if x == 'Loss' else 0)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = RandomForestClassifier(random_state=42)
        model.fit(X_train, y_train)

        joblib.dump(model, model_path)
        st.success("✅ تم تدريب النموذج وحفظه بنجاح!")
        return True
    except Exception as e:
        st.error(f"❌ حدث خطأ أثناء تدريب النموذج: {str(e)}")
        return False

# درّب النموذج إذا لم يكن موجودًا
if not os.path.exists(model_path):
    train_and_save_model()

# --- منطق الأسباب ---
def add_loss_reason(row):
    if row['V1'] == 0 and row['A1'] > 0:
        return '⚠️ فقد بسبب جهد صفر وتيار على V1'
    elif row['V2'] == 0 and row['A2'] > 0:
        return '⚠️ فقد بسبب جهد صفر وتيار على V2'
    elif row['V3'] == 0 and row['A3'] > 0:
        return '⚠️ فقد بسبب جهد صفر وتيار على V3'
    elif row['V1'] == 0 and row['A1'] == 0 and abs(row['A2'] - row['A3']) > 0.6 * max(row['A2'], row['A3']):
        return '⚠️ فقد بسبب عدم توازن التيار بين A2 و A3 مع جهد صفر على V1'
    elif row['V2'] == 0 and row['A2'] == 0 and abs(row['A1'] - row['A3']) > 0.6 * max(row['A1'], row['A3']):
        return '⚠️ فقد بسبب عدم توازن التيار بين A1 و A3 مع جهد صفر على V2'
    elif row['V3'] == 0 and row['A3'] == 0 and abs(row['A1'] - row['A2']) > 0.6 * max(row['A1'], row['A2']):
        return '⚠️ فقد بسبب عدم توازن التيار بين A1 و A2 مع جهد صفر على V3'
    else:
        return '✅ لا توجد حالة فقد مؤكدة'

# --- تحليل البيانات ---
def analyze_data(data):
    try:
        required_cols = {'V1','V2','V3','A1','A2','A3','Meter Number'}
        missing = required_cols - set(data.columns)
        if missing:
            st.error(f"⚠️ الملف ينقصه الأعمدة التالية: {', '.join(missing)}")
            return

        if not os.path.exists(model_path):
            st.error("⚠️ النموذج غير متوفر ولم يتمكن التطبيق من تدريبه. ارفع ملف التدريب أولًا.")
            return

        model = joblib.load(model_path)
        X = data[['V1', 'V2', 'V3', 'A1', 'A2', 'A3']]
        predictions = model.predict(X)

        data = data.copy()
        data['Predicted_Loss'] = predictions
        loss_data = data[data['Predicted_Loss'] == 1].copy()

        # إضافة أسباب الفقد
        loss_data['Reason'] = loss_data.apply(add_loss_reason, axis=1)

        # فرز الأولوية (إبقاء التحذيرات أولاً)
        high_priority_cases = loss_data[loss_data['Reason'].str.contains('⚠️')]

        st.subheader("📊 ملخص الحالات")
        st.info(f"🔍 عدد حالات الفقد المكتشفة: **{len(loss_data)}**")
        st.warning(f"🚨 عدد حالات الفقد ذات الأولوية العالية: **{len(high_priority_cases)}**")

        st.subheader("📋 جميع حالات الفقد المكتشفة")
        st.dataframe(loss_data, use_container_width=True)

        st.subheader("⚠️ حالات الفقد ذات الأولوية العالية")
        st.dataframe(high_priority_cases, use_container_width=True)

        # ملف Excel للتنزيل
        output_loss = BytesIO()
        with pd.ExcelWriter(output_loss, engine='xlsxwriter') as writer:
            loss_data.to_excel(writer, index=False)
        output_loss.seek(0)

        st.download_button(
            "📥 تحميل جميع حالات الفقد",
            data=output_loss,
            file_name="all_loss_cases.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

    except Exception as e:
        st.error(f"❌ حدث خطأ أثناء تحليل البيانات: {str(e)}")

# ---------------- واجهة التطبيق ----------------
st.sidebar.title("⚙️ إعدادات التطبيق")
st.sidebar.markdown("🔍 استخدم الخيارات أدناه لتحليل بيانات الفقد المحتملة.")

st.title("🔌 التنبؤ بحالات الفقد المحتملة")
st.markdown("### 📢 تحليل بيانات العدادات للكشف عن الفقد المحتمل")

# تحميل نموذج البيانات (قالب)
if os.path.exists(data_frame_template_path):
    with open(data_frame_template_path, 'rb') as template_file:
        template_data = template_file.read()
    st.sidebar.download_button("📥 تحميل قالب البيانات", data=template_data, file_name="The_data_frame_file_to_be_analyzed.xlsx")
else:
    st.sidebar.warning("⚠️ قالب البيانات غير متوفر! تأكد من رفعه إلى GitHub.")

st.header("📊 تحليل البيانات")
uploaded_analyze_file = st.file_uploader("📤 قم برفع ملف البيانات للتحليل (Excel)", type=["xlsx"])
if uploaded_analyze_file is not None:
    try:
        analyze_data_df = pd.read_excel(uploaded_analyze_file)
        analyze_data(analyze_data_df)
    except Exception as e:
        st.error(f"❌ خطأ في تحميل الملف: {str(e)}")

st.markdown("---")
st.markdown("### 👨‍💻 **المطور: مشهور العباس**")
st.markdown("📅 **تاريخ التحديث:** 2025-03-08")
