import os
import streamlit as st
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from io import BytesIO

# ===================== إعداد الصفحة =====================
st.set_page_config(page_title="نظام تحليل الفقد الذكي", page_icon="⚡", layout="wide")
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

# ===================== مسارات النظام =====================
if os.name == "nt":
    model_folder = "C:\\asd6"
else:
    model_folder = "asd6"

if not os.path.exists(model_folder):
    os.makedirs(model_folder)

model_path = os.path.join(model_folder, 'ASD6.pkl')
data_frame_template_path = 'The data frame file to be analyzed.xlsx'

# ===================== تدريب النموذج =====================
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

        model = RandomForestClassifier(
            n_estimators=500,
            max_depth=12,
            min_samples_split=4,
            class_weight="balanced",
            random_state=42
        )
        model.fit(X_train, y_train)
        joblib.dump(model, model_path)
        st.success("✅ تم تدريب النموذج وحفظه بنجاح!")
        return True

    except Exception as e:
        st.error(f"❌ حدث خطأ أثناء تدريب النموذج: {str(e)}")
        return False

if not os.path.exists(model_path):
    train_and_save_model()

# ===================== دالة التفسير =====================
def add_loss_reason(row, voltage_threshold=5, imbalance_ratio=0.5):
    V1, V2, V3 = row['V1'], row['V2'], row['V3']
    A1, A2, A3 = row['A1'], row['A2'], row['A3']

    # جهد منخفض مع تيار ظاهر
    if V1 < voltage_threshold and A1 > 0.2:
        return "⚠️ فقد بسبب جهد منخفض جدًا وتيار على V1"
    elif V2 < voltage_threshold and A2 > 0.2:
        return "⚠️ فقد بسبب جهد منخفض جدًا وتيار على V2"
    elif V3 < voltage_threshold and A3 > 0.2:
        return "⚠️ فقد بسبب جهد منخفض جدًا وتيار على V3"

    # عدم توازن التيارات (احتمال جمبر)
    max_current = max(A1, A2, A3)
    min_current = min(A1, A2, A3)
    if max_current > 0 and (max_current - min_current) / max_current > imbalance_ratio:
        dominant_phase = ["A1", "A2", "A3"][[A1, A2, A3].index(max_current)]
        return f"⚠️ عدم توازن كبير في التيارات - التيار الأعلى في {dominant_phase} (اشتباه جمبر بين الفازات)"

    # فرق الجهد بين الفازات
    max_voltage = max(V1, V2, V3)
    min_voltage = min(V1, V2, V3)
    if (max_voltage - min_voltage) / max_voltage > 0.15:
        return "⚠️ فرق جهد بين الفازات أعلى من 15% - احتمال خلل في التوصيل أو جمبر جزئي"

    return "✅ لا توجد حالة فقد مؤكدة"

# ===================== حساب درجة الخطورة =====================
def calculate_severity(row):
    V1, V2, V3 = row['V1'], row['V2'], row['V3']
    A1, A2, A3 = row['A1'], row['A2'], row['A3']

    # حساب شدة عدم توازن التيار
    max_a, min_a = max(A1, A2, A3), min(A1, A2, A3)
    current_imbalance = 0 if max_a == 0 else (max_a - min_a) / max_a

    # فرق الجهد بين الفازات
    max_v, min_v = max(V1, V2, V3), min(V1, V2, V3)
    voltage_diff = 0 if max_v == 0 else (max_v - min_v) / max_v

    # النقاط المجمعة
    score = current_imbalance
    if V1 < 5 or V2 < 5 or V3 < 5:
        score += 0.5
    if voltage_diff > 0.15:
        score += 0.3
    return round(score, 2)

def classify_priority(row):
    if row['Severity_Score'] >= 0.7:
        return "🔴 عالي جدًا"
    elif row['Severity_Score'] >= 0.4:
        return "🟠 متوسط"
    elif row['Severity_Score'] >= 0.2:
        return "🟡 منخفض"
    else:
        return "🟢 سليم"

# ===================== التحليل الرئيسي =====================
def analyze_data(data):
    try:
        required_cols = {'V1','V2','V3','A1','A2','A3','Meter Number'}
        missing = required_cols - set(data.columns)
        if missing:
            st.error(f"⚠️ الملف ينقصه الأعمدة التالية: {', '.join(missing)}")
            return

        model = joblib.load(model_path)
        X = data[['V1', 'V2', 'V3', 'A1', 'A2', 'A3']]
        predictions = model.predict(X)
        data['Predicted_Loss'] = predictions
        data['Reason'] = data.apply(add_loss_reason, axis=1)
        data['Severity_Score'] = data.apply(calculate_severity, axis=1)
        data['Priority'] = data.apply(classify_priority, axis=1)

        # تصنيف الحالات
        def classify_case(row):
            if row['Predicted_Loss'] == 1 and "⚠️" in row['Reason']:
                return "📊 فاقد مؤكد (النموذج + المحددات)"
            elif row['Predicted_Loss'] == 1:
                return "🤖 فاقد مكتشف من النموذج فقط"
            elif row['Predicted_Loss'] == 0 and "⚠️" in row['Reason']:
                return "🧠 حالة تنطبق عليها المحددات ولم يكتشفها النموذج"
            else:
                return "✅ سليم"
        data['Case_Type'] = data.apply(classify_case, axis=1)

        # استخراج المجموعات
        detected_loss = data[data['Predicted_Loss'] == 1]
        high_priority = detected_loss[detected_loss['Reason'].str.contains('⚠️')]
        logical_only = data[data['Case_Type'].str.contains("المحددات")]
        high_impact = data[data['Severity_Score'] >= 0.7].sort_values(by='Severity_Score', ascending=False)

        # ========== 📊 الإحصائيات العامة ==========
        total_cases = len(data)
        detected_count = len(detected_loss)
        logical_count = len(logical_only)
        high_impact_count = len(high_impact)
        estimated_loss_ratio = round(((detected_count + logical_count) / total_cases) * 100, 2)

        st.markdown("## 📈 إحصائيات التحليل")
        col1, col2, col3, col4, col5 = st.columns(5)
        col1.metric("📦 إجمالي العدادات", total_cases)
        col2.metric("🤖 حالات مكتشفة بالنموذج", detected_count)
        col3.metric("🧠 حالات هندسية فقط", logical_count)
        col4.metric("🔴 عالية التأثير", high_impact_count)
        col5.metric("📊 نسبة الفقد التقديرية", f"{estimated_loss_ratio}%")

        # ========== عرض الجداول ==========
        st.markdown("---")
        st.subheader("📊 جميع حالات الفقد المكتشفة من النموذج")
        st.dataframe(detected_loss)

        st.subheader("🚨 حالات الفقد ذات الأولوية العالية")
        st.dataframe(high_priority)

        st.subheader("🧠 حالات تنطبق عليها المحددات ولم يكتشفها النموذج")
        st.dataframe(logical_only)

        st.subheader("🔴 الحالات الأعلى تأثيرًا وخطورة")
        st.dataframe(high_impact)

        # تصدير للتحميل
        output = BytesIO()
        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
            detected_loss.to_excel(writer, sheet_name="Model_Detected", index=False)
            high_priority.to_excel(writer, sheet_name="High_Priority", index=False)
            logical_only.to_excel(writer, sheet_name="Logical_Only", index=False)
            high_impact.to_excel(writer, sheet_name="High_Impact", index=False)
        output.seek(0)

        st.download_button(
            "📥 تحميل تقرير شامل (Excel)",
            data=output,
            file_name="loss_analysis_report.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

    except Exception as e:
        st.error(f"❌ حدث خطأ أثناء تحليل البيانات: {str(e)}")

# ===================== واجهة التطبيق =====================
st.sidebar.title("⚙️ إعدادات التطبيق")
st.sidebar.markdown("🔍 استخدم الخيارات أدناه لتحليل بيانات الفقد المحتملة.")

st.title("⚡ نظام ذكي لتحليل واكتشاف حالات الفقد")
st.markdown("### يجمع بين النموذج التنبؤي والتحليل الهندسي لتحديد الحالات المؤكدة والأكثر تأثيرًا.")

# تحميل القالب
if os.path.exists(data_frame_template_path):
    with open(data_frame_template_path, 'rb') as template_file:
        template_data = template_file.read()
    st.sidebar.download_button("📥 تحميل قالب البيانات", data=template_data, file_name="The_data_frame_file_to_be_analyzed.xlsx")
else:
    st.sidebar.warning("⚠️ قالب البيانات غير متوفر! تأكد من رفعه إلى GitHub.")

# رفع البيانات
st.header("📤 تحليل ملف البيانات")
uploaded_file = st.file_uploader("قم برفع ملف Excel يحتوي على بيانات العدادات", type=["xlsx"])
if uploaded_file is not None:
    try:
        df = pd.read_excel(uploaded_file)
        analyze_data(df)
    except Exception as e:
        st.error(f"❌ خطأ في تحميل الملف: {str(e)}")

st.markdown("---")
st.markdown("### 👨‍💻 **المطور: مشهور العباس**")
st.markdown("📅 **آخر تحديث:** 2025-10-21")
