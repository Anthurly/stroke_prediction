import streamlit as st
import joblib
import pandas as pd

# Tải pipeline đã huấn luyện (có cả preprocessing + SMOTE + model)
model = joblib.load('stacking_stroke_model.pkl')

st.set_page_config(page_title="Dự đoán đột quỵ", layout="centered")
st.title(" Dự đoán nguy cơ đột quỵ")

# Giao diện nhập liệu
st.header(" Nhập thông tin bệnh nhân:")

gender = st.selectbox("Giới tính", ["Nam", "Nữ"])
age = st.slider("Tuổi", 0, 120, 50)
hypertension = st.radio("Tăng huyết áp?", ["Không", "Có"])
heart_disease = st.radio("Bệnh tim?", ["Không", "Có"])
avg_glucose_level = st.number_input("Chỉ số đường huyết trung bình", min_value=50.0, max_value=300.0, value=100.0)
bmi = st.number_input("Chỉ số BMI", min_value=10.0, max_value=100.0, value=25.0)
smoking_status = st.selectbox("Tình trạng hút thuốc", ["Chưa từng hút thuốc", "Đã từng hút thuốc trước đây", "Đang hút thuốc"])

# Tạo dataframe đầu vào
input_data = pd.DataFrame({
    'gender': [gender],
    'age': [age],
    'hypertension': [1 if hypertension == "Có" else 0],
    'heart_disease': [1 if heart_disease == "Có" else 0],
    'avg_glucose_level': [avg_glucose_level],
    'bmi': [bmi],
    'smoking_status': [smoking_status]
})

# Thêm các đặc trưng mới giống như trong quá trình huấn luyện
input_data['age_group'] = pd.cut(input_data['age'], bins=[0, 50, 80, 120], labels=['Người trẻ', 'Trung niên', 'Người lớn tuổi'], right=False)
input_data['bmi_category'] = pd.cut(input_data['bmi'], bins=[0, 18.5, 25, 30, 35, 40, 100],
                                    labels=['Gầy', 'Cân nặng bình thường', 'Thừa cân', 'Béo phì độ 1', 'Béo phì độ 2', 'Béo phì độ 3'], right=False)
input_data['age_hypertension'] = input_data['age'] * input_data['hypertension']
input_data['glucose_category'] = pd.cut(input_data['avg_glucose_level'],
                                        bins=[0, 70, 85, 100, 110, 126, 140, 300],
                                        labels=['Hạ đường huyết', 'Bình thường thấp', 'Bình thường', 'Cao nhẹ',
                                                'Tiền tiểu đường', 'Gần tiểu đường', 'Tiểu đường'], right=False)

# Dự đoán
if st.button("🔍 Dự đoán nguy cơ đột quỵ"):
    try:
        proba = model.predict_proba(input_data)[:, 1][0]
        threshold = 0.3
        prediction = int(proba >= threshold)

        st.subheader(" Kết quả:")
        st.write(f" Xác suất bị đột quỵ: **{proba:.2%}**")

        if prediction == 1:
            st.error(" Cảnh báo: Nguy cơ cao bị đột quỵ!")
        else:
            st.success(" Nguy cơ thấp. Tiếp tục theo dõi sức khỏe định kỳ.")
    except Exception as e:
        st.error(f"Đã xảy ra lỗi khi dự đoán: {e}")
