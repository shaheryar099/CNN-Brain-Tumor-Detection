🧠 Brain Tumor Detection Using Deep Learning

This project is an AI-powered web application designed to assist in the early detection of brain tumors from MRI scan images. It uses a deep learning model based on the VGG16 architecture to classify MRI images into four categories: Glioma Tumor, Meningioma Tumor, Pituitary Tumor, and No Tumor.

The model is trained using transfer learning, leveraging pre-trained ImageNet weights to improve accuracy and reduce training time. Uploaded MRI images are processed in real time, and predictions are generated directly in memory without saving images to disk, ensuring better performance and data privacy.

The application is built using TensorFlow/Keras for model inference and Streamlit for the user interface, providing a clean, user-friendly experience. Users can upload an MRI image, view the prediction result along with confidence scores, and receive informative guidance about the detected condition.

This system is intended for educational and research purposes and demonstrates how deep learning can support medical image analysis. It is not a replacement for professional medical diagnosis but can serve as a decision-support tool for clinicians and learners.

🔑 Key Features

MRI-based brain tumor classification

Deep learning with VGG16 (Transfer Learning)

Real-time prediction without image storage

User-friendly web interface

Confidence score and informative medical suggestions

🛠️ Technologies Used

Python

TensorFlow / Keras

VGG16 (CNN)

Streamlit

NumPy & PIL
