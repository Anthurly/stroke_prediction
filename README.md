# Dự đoán Đột Quỵ (Stroke Prediction)

Dự án sử dụng mô hình học máy **Stacking** để dự đoán khả năng đột quỵ dựa trên dữ liệu sức khỏe. Giao diện được xây bằng **Streamlit**.

## File trong dự án

- `app.py`: Chạy giao diện dự đoán
- `healthcare-dataset-stroke-data.csv`: Dữ liệu sức khỏe
- `stacking_stroke_model.pkl`: Mô hình đã huấn luyện
- `requirements.txt`: Thư viện cần cài

## Cách chạy

```bash
pip install -r requirements.txt
streamlit run app.py
