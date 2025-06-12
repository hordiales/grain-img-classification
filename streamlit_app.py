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
    try:
        model = YOLO('best.pt')  # Asegúrate que el modelo está en el mismo directorio
        #model = solutions.Inference(model="best.pt",)  # you can use any model that Ultralytics support, i.e. YOLO11, YOLOv10)
        st.write("Modelo cargado correctamente")
        #st.write(f"Tipo de modelo: {type(model)}")
        #st.write(f"Modelo path: {model.ckpt_path}")
        st.write(f"Clases detectadas: {model.names}")
        return model
    except Exception as e:
        st.error(f"Error al cargar el modelo: {e}")
        raise

model = load_model()

# Subir la imagen
uploaded_file = st.file_uploader("Subí una imagen de grano de soja", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Imagen cargada", use_container_width=True)

    # Realizar la predicción
    results = model.predict(image)
    #st.write(f"**Predicción:** {results}")

    # Modelo de clasificación
    # Asumimos clasificación binaria
#    for result in results:
#        # Obtener la clase predicha
#        class_id = int(result.probs.top1)
#        confidence = float(result.probs.top1conf)
#
#        # Mapear las clases
#        clases = {0: "Apto", 1: "Dañado"}
#        prediccion = clases.get(class_id, "Desconocido")
#
#        st.write(f"**Predicción:** {prediccion}")
#        st.write(f"**Confianza:** {confidence*100:.2f}%")

    # Modelo de detección
    for result in results:
        boxes = result.boxes  # todas las detecciones
        
        if boxes is None or len(boxes) == 0:
            st.write("No se detectaron granos en la imagen.")
        else:
            for box in boxes:
                class_id = int(box.cls[0].item())
                confidence = float(box.conf[0].item())

                clases = {0: "Apto", 1: "Dañado"}  # depende de cómo lo entrenaste
                prediccion = clases.get(class_id, "Desconocido")

                st.write(f"**Predicción:** {prediccion}")
                st.write(f"**Confianza:** {confidence*100:.2f}%")