import streamlit as st
import cv2
import numpy as np
from recognizer import FaceRecognizer

st.set_page_config(page_title="Person 2 - Face Recognition", page_icon="🔐")

@st.cache_resource
def load_recognizer():
    return FaceRecognizer()

recognizer = load_recognizer()

st.title("🔐 Person 2 - Face Detection & Recognition")
st.write("Real face detection, registration, and authorized/unknown recognition.")

tab1, tab2 = st.tabs(["📸 Recognition", "👤 Registration"])

with tab1:
    st.subheader("Recognize a person")
    image_file = st.camera_input("Take a photo")

    if image_file is not None:
        data = np.asarray(bytearray(image_file.read()), dtype=np.uint8)
        frame = cv2.imdecode(data, cv2.IMREAD_COLOR)

        if frame is None:
            st.error("Could not read the camera image.")
        else:
            result = recognizer.recognize(frame)
            st.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), caption="Captured image")

            if not result.get("face_detected", False):
                st.warning(result.get("message", "No face detected."))
            elif result.get("face_count", 1) != 1:
                st.warning(result.get("message", "Please show exactly one face."))
            elif result.get("recognized", False):
                st.success("✅ AUTHORIZED PERSON RECOGNIZED")
                st.write("Name:", result.get("name", "Unknown"))
                st.write("Distance:", round(result.get("distance", 0), 4))
                st.write("Similarity:", f"{result.get('confidence', 0):.2f}%")
            else:
                st.error("❌ UNKNOWN PERSON")
                st.write("This face is not registered.")
                if result.get("distance") is not None:
                    st.write("Distance:", round(result["distance"], 4))

with tab2:
    st.subheader("Register an authorized person")
    name = st.text_input("Person name")
    number = st.number_input("Number of samples", min_value=1, max_value=5, value=3)

    captured_images = []

    for i in range(int(number)):
        image = st.camera_input(f"Capture sample {i + 1}", key=f"registration_camera_{i}")
        if image is not None:
            data = np.asarray(bytearray(image.read()), dtype=np.uint8)
            frame = cv2.imdecode(data, cv2.IMREAD_COLOR)
            if frame is not None:
                captured_images.append(frame)

    if st.button("Register person"):
        if not name.strip():
            st.error("Please enter a name.")
        elif not captured_images:
            st.error("Please capture at least one sample.")
        else:
            result = recognizer.enroll(name.strip(), captured_images)
            if result.get("success"):
                st.success(result.get("message", "Registration successful."))
            else:
                st.error(result.get("message", "Registration failed."))
            st.write("Samples saved:", result.get("samples_saved", 0))
            st.write("Samples rejected:", result.get("samples_rejected", 0))

st.divider()
st.subheader("Registered Users")

users = recognizer.db.list_users()

if not users:
    st.info("No registered users yet.")
else:
    for user in users:
        st.write(f"👤 {user} — {recognizer.db.count_samples(user)} sample(s)")

st.caption(f"Recognition threshold: {recognizer.threshold}")
