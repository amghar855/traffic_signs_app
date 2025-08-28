import streamlit as st
import tempfile
from PIL import Image
from ultralytics import YOLO
import imageio

# تحميل الموديل
model = YOLO("best.pt")

st.title("🚦 Traffic Signs Classifier")

# اختيار الوضع
choice = st.sidebar.radio("Choose one:", ["Picture", "Video", "Camera"])

# ---- 1. صورة ----
if choice == "Picture":
    uploaded_file = st.file_uploader("📷 Upload an image", type=["jpg", "png", "jpeg"])
    if uploaded_file:
        img = Image.open(uploaded_file)
        st.image(img, caption="The uploaded image", use_column_width=True)

        results = model.predict(img)
        st.subheader("✅ Results")
        st.write(results[0].names)
        st.image(results[0].plot(), caption="Classification", use_column_width=True)

# ---- 2. فيديو ----
elif choice == "Video":
    uploaded_video = st.file_uploader("🎥 Upload a video", type=["mp4", "mov", "avi"])
    if uploaded_video:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_video.read())

        stframe = st.empty()
        vid = imageio.get_reader(tfile.name)

        for frame in vid:
            results = model.predict(frame)
            annotated_frame = results[0].plot()

            stframe.image(annotated_frame, channels="BGR")

# ---- 3. كاميرا (غير مدعومة على Streamlit Cloud) ----
elif choice == "Camera":
    st.warning("⚠️ Camera is not supported on Streamlit Cloud. Please run locally on your PC.")
