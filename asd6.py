import os
import streamlit as st
import pandas as pd
import numpy as np
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
os.makedirs(model_folder, exist_ok=True)

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

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

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
        st.error(f"❌ خطأ أثناء تدريب النموذج: {str(e)}")
        return False

if not os.path.exists(model_path):
    train_and_save_model()

# ===================== ثوابت هندسية (قابلة للضبط من الشريط الجانبي) =====================
st.sidebar.header("⚙️ عتبات التحليل")
I_abs_min = st.sidebar.number_input("حد التيار المهمل (A)", 0.1, 10.0, 0.5, 0.1)
I_rel_min = st.sidebar.slider("حد التيار النسبي من المتوسط", 0.0, 0.5, 0.10, 0.01)
pair_similarity = st.sidebar.slider("تقارب تياري الفازتين المستخدمتين", 0.05, 0.5, 0.18, 0.01)
V_zero_thr = st.sidebar.slider("عتبة الجهد ≈ صفر (V)", 0, 50, 20, 1)
V_ok_min = st.sidebar.slider("أدنى جهد طبيعي (V)", 150, 240, 180, 1)
IUF_alarm = st.sidebar.slider("حد عدم توازن التيار IUF", 0.1, 1.0, 0.40, 0.01)
VUF_alarm = st.sidebar.slider("حد عدم توازن الجهد (VUF≈)", 0.05, 0.5, 0.15, 0.01)

# ===================== أدوات مساعدة =====================
def avg(x, eps=1e-9):
    return max(np.mean(x), eps)

def iuf(a1, a2, a3):
    A = np.array([a1, a2, a3], dtype=float)
    Abar = avg(A)
    return float(np.max(np.abs(A - Abar)) / Abar)

def vuf(v1, v2, v3):
    V = np.array([v1, v2, v3], dtype=float)
    Vmax, Vmin = float(np.max(V)), float(np.min(V))
    return 0.0 if Vmax == 0 else (Vmax - Vmin) / Vmax

# ===================== كشف عداد ثنائي الفاز (صارم) =====================
def detect_two_phase_strict(row):
    V = [float(row['V1']), float(row['V2']), float(row['V3'])]
    A = [float(row['A1']), float(row['A2']), float(row['A3'])]
    names = ['A1', 'A2', 'A3']

    # 1) حدد فازة مرشحة كـ "غير مستخدمة"
    candidates = []
    for i in range(3):
        # مرشح إذا: (V≈0) أو (I≈0 مع V طبيعي)
        cond_v_zero = V[i] <= V_zero_thr
        cond_i_zero = (A[i] <= max(I_abs_min, I_rel_min * avg([A[(i+1)%3], A[(i+2)%3]]))) and (V[i] >= V_ok_min)
        if cond_v_zero or cond_i_zero:
            candidates.append(i)

    # يجب أن تكون فازة واحدة فقط مرشحة
    if len(candidates) != 1:
        return (False, None, None, None)

    unused = candidates[0]
    used = [j for j in range(3) if j != unused]
    a_used = [A[j] for j in used]

    # 2) **شرط مطلق**: الفازة غير المستخدمة **فعلاً** تيارها ضعيف (مطلقًا ونِسبيًا)
    A_unused_ok = A[unused] <= max(I_abs_min, I_rel_min * avg(a_used))

    # 3) الفازتان المستخدمتان متقاربتان وبقيم معقولة
    hi, lo = max(a_used), min(a_used)
    similar = (hi == 0 and lo == 0) or (hi > 0 and (hi - lo) / hi <= pair_similarity)
    strong_enough = (hi >= max(I_abs_min, I_rel_min * avg(a_used))) and (lo >= max(I_abs_min, I_rel_min * avg(a_used)))

    # 4) **حالة إلغاء صريحة**: إن كانت **الفازات الثلاث** تياراتها جميعًا أكبر من حد معقول ⇒ ليس ثنائي فاز
    all_significant = all(x >= max(I_abs_min, I_rel_min * avg(A)) for x in A)

    if A_unused_ok and similar and strong_enough and (not all_significant):
        unused_reason = "جهد غير موصول (V≈0)" if V[unused] <= V_zero_thr else "تيار ≈ صفر مع جهد طبيعي"
        return (True, names[unused], f"{names[used[0]]}+{names[used[1]]}", unused_reason)

    return (False, None, None, None)

# ===================== التفسير الهندسي =====================
def reason_engine(row):
    V1, V2, V3 = float(row['V1']), float(row['V2']), float(row['V3'])
    A1, A2, A3 = float(row['A1']), float(row['A2']), float(row['A3'])
    total_I = A1 + A2 + A3

    # حمولة عامة منخفضة = نتجاهل الضوضاء
    if total_I < 3 * I_abs_min:
        return "ℹ️ تحميل منخفض (لا إنذارات)"

    # جهد منخفض مع تيار
    for p, Vp, Ap in [("A1",V1,A1),("A2",V2,A2),("A3",V3,A3)]:
        if Vp < 5 and Ap >= I_abs_min:
            return f"⚠️ جهد منخفض جدًا مع تيار على {p}"

    # عدم توازن الجهد
    VUF = vuf(V1, V2, V3)
    if VUF >= VUF_alarm:
        return "⚠️ فرق جهد كبير بين الفازات"

    # عدم توازن التيار
    IUF = iuf(A1, A2, A3)
    if IUF >= IUF_alarm:
        # تحديد الفازة المسيطرة
        mx = max(A1, A2, A3)
        p = ["A1","A2","A3"][[A1,A2,A3].index(mx)]
        return f"⚠️ عدم توازن كبير في التيارات (الأعلى {p})"

    return "✅ لا توجد حالة فقد مؤكدة"

# ===================== حساب الخطورة =====================
def severity_score(row):
    if row.get('Two_Phase_Meter', False):
        return 0.0
    V1, V2, V3 = float(row['V1']), float(row['V2']), float(row['V3'])
    A1, A2, A3 = float(row['A1']), float(row['A2']), float(row['A3'])
    s = 0.0
    s += 0.6 * iuf(A1, A2, A3)
    s += 0.3 * vuf(V1, V2, V3)
    lowVflag = int((V1 < 5 and A1 >= I_abs_min) or (V2 < 5 and A2 >= I_abs_min) or (V3 < 5 and A3 >= I_abs_min))
    s += 0.1 * lowVflag
    return round(min(max(s, 0.0), 1.0), 2)

def priority_label(s):
    if s >= 0.7: return "🔴 عالي جدًا"
    if s >= 0.4: return "🟠 متوسط"
    if s >= 0.2: return "🟡 منخفض"
    return "🟢 سليم"

# ===================== التحليل الرئيسي =====================
def analyze_data(df):
    try:
        required = {'V1','V2','V3','A1','A2','A3','Meter Number'}
        missing = required - set(df.columns)
        if missing:
            st.error(f"⚠️ الملف ينقصه الأعمدة التالية: {', '.join(missing)}")
            return

        model = joblib.load(model_path)

        X = df[['V1','V2','V3','A1','A2','A3']]
        df = df.copy()
        df['Predicted_Loss'] = model.predict(X)

        # كاشف ثنائي الفاز الصارم
        tp_cols = df.apply(
            lambda r: pd.Series(detect_two_phase_strict(r),
                                index=['Two_Phase_Meter','Unused_Phase','Two_Phase_Pair','Unused_Reason']),
            axis=1
        )
        df[['Two_Phase_Meter','Unused_Phase','Two_Phase_Pair','Unused_Reason']] = tp_cols

        # التفسير الهندسي
        df['Reason'] = df.apply(reason_engine, axis=1)
        df.loc[df['Two_Phase_Meter'] == True, 'Reason'] = df.loc[df['Two_Phase_Meter'] == True].apply(
            lambda r: f"ℹ️ عداد ثنائي الفاز (مستثنى): غير مستخدم {r['Unused_Phase']} ({r['Unused_Reason']}), مستخدم {r['Two_Phase_Pair']}",
            axis=1
        )

        # الخطورة والأولوية
        df['Severity_Score'] = df.apply(severity_score, axis=1)
        df['Priority'] = df['Severity_Score'].apply(priority_label)

        # نوع الحالة
        def classify_case(r):
            if r['Two_Phase_Meter']: return "ℹ️ عداد ثنائي الفاز (مستثنى)"
            if r['Predicted_Loss']==1 and "⚠️" in r['Reason']: return "📊 فاقد مؤكد (نموذج+محددات)"
            if r['Predicted_Loss']==1: return "🤖 فاقد (النموذج فقط)"
            if r['Predicted_Loss']==0 and "⚠️" in r['Reason']: return "🧠 فاقد هندسي فقط"
            return "✅ سليم"
        df['Case_Type'] = df.apply(classify_case, axis=1)

        # ===== إزالة التكرار لكل عداد (نأخذ الأعلى خطورة) =====
        df = df.sort_values(['Meter Number','Severity_Score'], ascending=[True, False])\
               .drop_duplicates(subset=['Meter Number'], keep='first')

        # مجموعات العرض
        detected_loss = df[df['Predicted_Loss'] == 1]
        high_priority = detected_loss[detected_loss['Reason'].str.contains('⚠️')]
        logical_only = df[df['Case_Type'].str.contains("هندسي")]
        two_phase_only = df[df['Two_Phase_Meter'] == True]
        high_impact = df[(df['Severity_Score'] >= 0.7) & (~df['Two_Phase_Meter'])]\
                        .sort_values(by='Severity_Score', ascending=False)

        # إحصائيات
        total = len(df)
        tp_count = len(two_phase_only)
        effective_total = max(total - tp_count, 1)
        detected_count = len(detected_loss)
        logical_count = len(logical_only)
        high_impact_count = len(high_impact)
        loss_ratio = round(((detected_count + logical_count) / effective_total) * 100, 2)

        st.markdown("## 📈 إحصائيات التحليل")
        c1, c2, c3, c4, c5, c6 = st.columns(6)
        c1.metric("عدد العدادات", total)
        c2.metric("ثنائي فاز (مستثنى)", tp_count)
        c3.metric("فاقد بالنموذج", detected_count)
        c4.metric("فاقد هندسي فقط", logical_count)
        c5.metric("عالي التأثير", high_impact_count)
        c6.metric("نسبة الفقد (بعد الاستثناء)", f"{loss_ratio}%")

        st.markdown("---")
        st.subheader("📊 جميع حالات الفقد المكتشفة من النموذج")
        st.dataframe(detected_loss)

        st.subheader("🚨 حالات الفقد ذات الأولوية العالية")
        st.dataframe(high_priority)

        st.subheader("🧠 حالات تنطبق عليها المحددات ولم يكتشفها النموذج")
        st.dataframe(logical_only)

        st.subheader("🔴 الحالات الأعلى تأثيرًا وخطورة")
        st.dataframe(high_impact)

        st.subheader("ℹ️ عدادات ثنائية الفاز (مستثناة)")
        st.dataframe(two_phase_only)

        # تقرير Excel بنزع التكرار
        output = BytesIO()
        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
            detected_loss.to_excel(writer, sheet_name="Model_Detected", index=False)
            high_priority.to_excel(writer, sheet_name="High_Priority", index=False)
            logical_only.to_excel(writer, sheet_name="Logical_Only", index=False)
            high_impact.to_excel(writer, sheet_name="High_Impact", index=False)
            two_phase_only.to_excel(writer, sheet_name="Two_Phase_Meters", index=False)
        output.seek(0)

        st.download_button(
            "📥 تحميل تقرير شامل (Excel)",
            data=output,
            file_name="loss_analysis_report.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

    except Exception as e:
        st.error(f"❌ خطأ أثناء تحليل البيانات: {str(e)}")

# ===================== واجهة التطبيق =====================
st.sidebar.title("🧪 التحليل")
st.title("⚡ نظام ذكي لتحليل واكتشاف حالات الفقد")
st.markdown("يضبط العتبات من الشريط الجانبي ويمنع تصنيف ثنائي فاز إذا كان هناك حمل ملحوظ على الفازات الثلاث.")

# تحميل القالب
if os.path.exists(data_frame_template_path):
    with open(data_frame_template_path, 'rb') as f:
        st.sidebar.download_button("📥 تحميل قالب البيانات", data=f.read(),
                                   file_name="The_data_frame_file_to_be_analyzed.xlsx")
else:
    st.sidebar.warning("⚠️ قالب البيانات غير متوفر!")

# رفع وتحليل
st.header("📤 تحليل ملف Excel")
file = st.file_uploader("ارفع الملف", type=["xlsx"])
if file is not None:
    try:
        df = pd.read_excel(file)
        analyze_data(df)
    except Exception as e:
        st.error(f"❌ خطأ في تحميل الملف: {str(e)}")

st.markdown("---")
st.markdown("👨‍💻 المطور: مشهور العباس — آخر تحديث اليوم")
