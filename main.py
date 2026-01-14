import tensorflow as tf
from tensorflow import keras
from keras.applications import VGG16
from keras.models import Sequential
from keras.layers import Input, Dense, Dropout, BatchNormalization, GlobalAveragePooling2D, Flatten
from keras.optimizers import Adam
from PIL import Image
import streamlit as st
import numpy as np
import os

IMAGE_SIZE = 128
CLASS_NAMES = ['Glioma Tumor', 'Meningioma Tumor', 'No Tumor', 'Pituitary Tumor']


#Model Building
def build_model_architecture():
    base_model1 = VGG16(weights='imagenet', include_top=False, input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3))

    for layer in base_model1.layers[:-6]:
        layer.trainable = False

    model1 = Sequential()
    model1.add(Input(shape=(IMAGE_SIZE, IMAGE_SIZE, 3)))
    model1.add(base_model1)
    model1.add(GlobalAveragePooling2D())

    model1.add(BatchNormalization())
    model1.add(Dense(256, activation='relu'))
    model1.add(Dropout(0.4))

    model1.add(BatchNormalization())
    model1.add(Dense(128, activation='relu'))
    model1.add(Dropout(0.3))

    model1.add(Dense(4, activation='softmax'))
    
    model1.compile(
        optimizer=Adam(learning_rate=1e-4),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model1


model = build_model_architecture()
if os.path.exists("model.weights.h5"):
    try:
        model.load_weights("model.weights.h5")
        st.sidebar.success("✅ Model weights loaded successfully")
    except Exception as e:
        st.sidebar.error(f"❌ Error loading weights: {str(e)[:100]}")
        st.sidebar.info("Using base model without pre-trained weights")
else:
    st.sidebar.warning("⚠️ No pre-trained weights found. Using base model.")


#Prediction
def predict_image_pil(pil_image):
    img = pil_image.resize((IMAGE_SIZE, IMAGE_SIZE))
    img = img.convert("RGB")

    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array, verbose=0)
    class_index = np.argmax(prediction)
    confidence = np.max(prediction) * 100

    return class_index, confidence, prediction[0]


#Streamlit Code
st.set_page_config(
    page_title="Brain Tumor Detection",
    page_icon="🧠",
    layout="centered"
)

st.markdown(
    """
    <h1 style="text-align:center;">🧠 Brain Tumor Prediction</h1>
    <p style="text-align:center;">
    Upload an MRI brain scan to predict the tumor type
    </p>
    """,
    unsafe_allow_html=True
)

st.divider()

uploaded_file = st.file_uploader(
    "📂 Upload MRI Image (JPG, JPEG, PNG | Max 20MB)"
)

if uploaded_file:
    file_size_mb = uploaded_file.size / (1024 * 1024)

    if file_size_mb > 20:
        st.error("File size exceeds 20 MB")
    else:
        image = Image.open(uploaded_file)

        col1, col2 = st.columns(2)

        with col1:
            st.image(image, caption="Uploaded MRI", use_container_width=True)

        with col2:
            st.markdown("### 🔍 Prediction Result")

            if st.button("Predict Tumor Type"):
                with st.spinner("Analyzing MRI scan..."):
                    result, confidence, probs = predict_image_pil(image)

                st.success(CLASS_NAMES[result])
                st.metric("Confidence", f"{confidence:.2f}%")
                
                if CLASS_NAMES[result] == "Glioma Tumor":
                    st.caption('The scan shows features consistent with a glioma, a tumor arising from glial cells in the brain. Gliomas can vary in severity and growth rate. Early consultation with a neurologist or oncologist is strongly recommended for further imaging, biopsy, and treatment planning')
                    
                elif CLASS_NAMES[result] == "Meningioma Tumor":
                    st.caption('The MRI indicates characteristics commonly associated with a meningioma, a tumor that develops from the membranes covering the brain and spinal cord. Many meningiomas grow slowly, but medical review is important to assess size, location, and whether treatment or monitoring is required.')
                    
                elif CLASS_NAMES[result] == "Pituitary Tumor":
                    st.caption('The scan suggests the presence of a pituitary tumor, which may affect hormone production and bodily functions such as growth, metabolism, and vision. An endocrinologist or neurospecialist should be consulted for hormonal tests and detailed assessment.')
                    
                else:
                    st.caption('The MRI scan does not show visible signs of a brain tumor. This generally indicates a normal brain structure. If symptoms persist (such as headaches, vision issues, or dizziness), follow up with a medical professional for further evaluation.') 

else:
    st.info("⬆️ Please upload an MRI image to get started.")
