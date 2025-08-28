import streamlit as st
import tempfile
from PIL import Image
from ultralytics import YOLO
import imageio

model = YOLO("best.pt")

st.title("Traffic Signs Classifier")

# اختيار الوضع
choice = st.sidebar.radio("choose one :", ["pecture", "vedeo", "camera"])

# ---- 1. صورة ----
if choice == "pecture":
    uploaded_file = st.file_uploader("set a pecture", type=["jpg", "png", "jpeg"])
    if uploaded_file:
        img = Image.open(uploaded_file)
        st.image(img, caption="The uploaded image", use_column_width=True)

        results = model.predict(img)
        st.write("results")
        st.write(results[0].names)
        st.image(results[0].plot(), caption="classification", use_column_width=True)

# ---- 2. فيديو ----
elif choice == "vedeo":
    uploaded_video = st.file_uploader("set a vedeo", type=["mp4", "mov", "avi"])
    if uploaded_video:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_video.read())

        cap = cv2.VideoCapture(tfile.name)
        stframe = st.empty()

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # تنبؤ
            results = model.predict(frame)
            annotated_frame = results[0].plot()

            stframe.image(annotated_frame, channels="BGR")

        cap.release()

elif choice == "Camera":
    st.warning("Camera is not supported on Streamlit Cloud. Please run locally on your PC.")
