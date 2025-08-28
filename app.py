import streamlit as st
from ultralytics import YOLO
from PIL import Image

# تحميل الموديل
model = YOLO("best.pt")

st.title("🚦 Traffic Sign Detection")

# رفع صورة من المستخدم
uploaded_file = st.file_uploader("اختر صورة", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # عرض الصورة
    image = Image.open(uploaded_file)
    st.image(image, caption="📷 الصورة المدخلة", use_column_width=True)

    # التنبؤ
    results = model.predict(image)

    # عرض الصورة مع البوكسات
    res_plotted = results[0].plot()  # numpy array
    st.image(res_plotted, caption="📍 النتيجة", use_column_width=True)
