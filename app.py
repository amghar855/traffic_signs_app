import streamlit as st
from ultralytics import YOLO
from PIL import Image

# تحميل الموديل
model = YOLO("best.pt")

st.title("Traffic Sign Detection")

# رفع صورة من المستخدم
uploaded_file = st.file_uploader("choose image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # عرض الصورة
    image = Image.open(uploaded_file)
    st.image(image, caption="the uploaded image ", use_column_width=True)

    # التنبؤ
    results = model.predict(image)

    # عرض الصورة مع البوكسات
    res_plotted = results[0].plot()  # numpy array
    st.image(res_plotted, caption="result", use_column_width=True)
