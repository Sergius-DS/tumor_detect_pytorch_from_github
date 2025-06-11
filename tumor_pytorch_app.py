import streamlit as st
import torch
import torchvision.transforms as transforms
from torchvision import models
import requests
import base64
from PIL import Image
import os

# Path a tu imagen de fondo
background_image_path = "medical_laboratory.jpg"

def get_base64_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode()

def set_background(image_path):
    b64_image = get_base64_image(image_path)
    css = f"""
    <style>
    .stApp {{
        background-image: url("data:image/jpeg;base64,{b64_image}");
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
        background-attachment: fixed;
    }}
    .main-title {{
        background-color: rgba(255, 255, 255, 0.9);
        padding: 20px;
        border-radius: 10px;
        margin: 20px 0;
        text-align: center;
    }}
    .prediction-box {{
        background-color: rgba(255, 255, 255, 0.95);
        padding: 15px;
        border-radius: 8px;
        margin: 10px 0;
        border-left: 4px solid #1f77b4;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
        text-align: center;
    }}
    .stFileUploader > div {{
        background-color: rgba(255, 255, 255, 0.9);
        padding: 10px;
        border-radius: 8px;
    }}
    .stAlert {{
        background-color: rgba(255, 255, 255, 0.95) !important;
    }}
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)

set_background(background_image_path)

# Etiquetas de clases
class_labels = ["Healthy", "Tumor"]

@st.cache(allow_output_mutation=True)
def load_model():
    # Cargar la arquitectura
    model = models.resnet50(pretrained=False)
    num_ftrs = model.fc.in_features
    # Ajustar la capa final para salida binaria con Sigmoid
    model.fc = torch.nn.Sequential(
        torch.nn.Dropout(0.5),
        torch.nn.Linear(num_ftrs, 1),
        torch.nn.Sigmoid()
    )
    # Cargar pesos
    model.load_state_dict(torch.load('best_model_final.pth', map_location=torch.device('cpu')))
    model.eval()
    return model

# Carga del modelo
with st.spinner("Cargando modelo... Esto puede tardar un momento."):
    model = load_model()

# Transformaciones iguales a las de entrenamiento
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])

# Interfaz de usuario
st.markdown("""
<div class="main-title">
    <h1>🧠 Detección de Tumor Cerebral con Deep Learning 🔎</h1>
</div>
""", unsafe_allow_html=True)

col1, col2 = st.columns([1, 1])

with col1:
    uploaded_file = st.file_uploader("Elige una imagen...", type=["jpg", "jpeg", "png"])
    predict_button = st.button("Predecir")
    # Mantener la imagen cargada si ya se cargó
    if uploaded_file:
        if st.session_state.get('uploaded_image') != uploaded_file:
            st.session_state['uploaded_image'] = uploaded_file
            st.session_state['prediction'] = None
    elif st.session_state.get('uploaded_image'):
        uploaded_file = st.session_state['uploaded_image']
    else:
        uploaded_file = None

with col2:
    if uploaded_file:
        image = Image.open(uploaded_file).convert('RGB')
        st.image(image, caption='Imagen cargada.', width=240)

# Predicción
if predict_button and uploaded_file:
    image = Image.open(uploaded_file).convert('RGB')
    input_tensor = transform(image)
    input_batch = input_tensor.unsqueeze(0)  # Mini-batch

    with torch.no_grad():
        output = model(input_batch)
        prob = output.item()  # valor entre 0 y 1
        if prob > 0.5:
            predicted_class = "Tumor"
            confidence = prob
        else:
            predicted_class = "Healthy"
            confidence = 1 - prob

    # Guardar en estado de sesión
    st.session_state['prediction'] = {
        'class': predicted_class,
        'confidence': confidence
    }

# Mostrar resultado
if st.session_state.get('prediction'):
    pred = st.session_state['prediction']
    st.markdown(f"""
    <div class="prediction-box">
        <h3>Resultado de la Predicción:</h3>
        <p><strong>{pred['class']}</strong> con confianza <strong>{pred['confidence']*100:.2f}%</strong></p>
    </div>
    """, unsafe_allow_html=True)
