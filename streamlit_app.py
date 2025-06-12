import streamlit as st
from PIL import Image
#import torch
from ultralytics import YOLO
#from ultralytics import solutions
# recomendado acá, ref: https://docs.ultralytics.com/guides/streamlit-live-inference/#streamlit-application-code



#inf.inference()


# Título de la app
st.title("Clasificador de granos de soja")

# Cargar el modelo YOLOv8 entrenado
@st.cache_resource
def load_model():
    model = YOLO('best.pt')  # Asegúrate que el modelo está en el mismo directorio
    #model = solutions.Inference(model="best.pt",)  # you can use any model that Ultralytics support, i.e. YOLO11, YOLOv10)
    return model

model = load_model()

# Subir la imagen
uploaded_file = st.file_uploader("Subí una imagen de grano de soja", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Imagen cargada", use_column_width=True)

    # Realizar la predicción
    results = model.predict(image)
    st.write(f"**Predicción:** {results}")

    # Asumimos clasificación binaria
    #for result in results:
        # Obtener la clase predicha
        #class_id = int(result.probs.top1)
        #confidence = float(result.probs.top1conf)

        # Mapear las clases
     #   clases = {0: "Apto", 1: "Dañado"}
      #  prediccion = clases.get(class_id, "Desconocido")

       # st.write(f"**Predicción:** {prediccion}")
        #st.write(f"**Confianza:** {confidence*100:.2f}%")
