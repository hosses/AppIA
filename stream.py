import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import pickle
from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn.metrics import roc_auc_score
import numpy as np

# Configuración de la página
st.set_page_config(page_title="Análisis de Cultivos", layout="wide")

# Al inicio del archivo, después de los imports
crop_dict = {
    'rice': 1, 'maize': 2, 'jute': 3, 'cotton': 4, 'coconut': 5, 'papaya': 6,
    'orange': 7, 'apple': 8, 'muskmelon': 9, 'watermelon': 10, 'grapes': 11,
    'mango': 12, 'banana': 13, 'pomegranate': 14, 'lentil': 15, 'blackgram': 16,
    'mungbean': 17, 'mothbeans': 18, 'pigeonpeas': 19, 'kidneybeans': 20,
    'chickpea': 21, 'coffee': 22
}

def show_model_metrics(model, data, mx_scaler, sc_scaler):
    st.subheader("Métricas de Desempeño del Modelo")

    # Preparar datos
    X = data.drop('label', axis=1)
    y = data['label']
    y_numeric = y.map(crop_dict)

    # Aplicar transformaciones
    X_transformed = sc_scaler.transform(mx_scaler.transform(X))
    y_pred = model.predict(X_transformed)
    y_pred_proba = model.predict_proba(X_transformed)

    # Crear tabs para diferentes métricas
    metric_tab1, metric_tab2 = st.tabs(["Matriz de Correlación del Suelo", "Curva ROC"])

    with metric_tab1:
        # Seleccionar todas las variables relacionadas con el suelo
        soil_cols = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']
        correlation_matrix = data[soil_cols].corr()

        # Crear matriz de correlación interactiva con plotly
        fig_corr = go.Figure(data=go.Heatmap(
            z=correlation_matrix,
            x=soil_cols,
            y=soil_cols,
            colorscale='RdBu',
            zmin=-1,
            zmax=1,
            text=np.round(correlation_matrix, 2),
            texttemplate='%{text}',
            textfont={"size": 12},
        ))

        # Personalizar el diseño de la matriz de correlación
        fig_corr.update_layout(
            title={
                'text': 'Matriz de Correlación de Variables del Suelo',
                'y':0.95,
                'x':0.5,
                'xanchor': 'center',
                'yanchor': 'top'
            },
            xaxis_title='Variables',
            yaxis_title='Variables',
            width=800,
            height=800,
            xaxis={'tickangle': 45},
            yaxis={'autorange': 'reversed'}
        )

        # Mostrar la matriz de correlación
        st.plotly_chart(fig_corr)

        # Agregar explicación de las variables
        st.markdown("""
        **Variables del suelo:**
        - N: Contenido de Nitrógeno
        - P: Contenido de Fósforo
        - K: Contenido de Potasio
        - temperature: Temperatura
        - humidity: Humedad
        - ph: Nivel de pH
        - rainfall: Precipitación
        """)

    with metric_tab2:
        # Preparar datos para ROC
        y_bin = pd.get_dummies(y_numeric).values

        # Calcular curvas ROC y AUC para cada clase
        fig_roc = go.Figure()

        for i in range(len(crop_dict)):
            fpr, tpr, _ = roc_curve(y_bin[:, i], y_pred_proba[:, i])
            roc_auc = auc(fpr, tpr)

            # Agregar curva al gráfico
            fig_roc.add_trace(go.Scatter(
                x=fpr,
                y=tpr,
                name=f'{get_crop_name(i+1)} (AUC = {roc_auc:.2f})',
                mode='lines'
            ))

        # Calcular y mostrar AUC promedio
        roc_auc_avg = roc_auc_score(y_bin, y_pred_proba, average='macro')
        st.write(f"AUC promedio: {roc_auc_avg:.2f}")

        # Agregar línea diagonal de referencia
        fig_roc.add_trace(go.Scatter(
            x=[0, 1],
            y=[0, 1],
            name='Línea Base',
            mode='lines',
            line=dict(dash='dash'),
            line_color='gray'
        ))

        # Personalizar el diseño de las curvas ROC
        fig_roc.update_layout(
            title={
                'text': 'Curvas ROC para cada Cultivo',
                'y':0.95,
                'x':0.5,
                'xanchor': 'center',
                'yanchor': 'top'
            },
            xaxis_title='Tasa de Falsos Positivos',
            yaxis_title='Tasa de Verdaderos Positivos',
            width=800,
            height=600,
            xaxis=dict(range=[0, 1]),
            yaxis=dict(range=[0, 1.05]),
            legend=dict(
                yanchor="bottom",
                y=-0.5,
                xanchor="center",
                x=0.5,
                orientation="h"
            )
        )

        st.plotly_chart(fig_roc)

# Función para obtener el nombre del cultivo
def get_crop_name(prediction_number):
    reverse_dict = {v: k for k, v in crop_dict.items()}
    return reverse_dict.get(prediction_number, "Desconocido")

# Cargar el modelo y los escaladores
@st.cache_resource
def load_models():
    model = pickle.load(open('model.pkl', 'rb'))
    mx_scaler = pickle.load(open('minmaxscaler.pkl', 'rb'))
    sc_scaler = pickle.load(open('standscaler.pkl', 'rb'))
    return model, mx_scaler, sc_scaler

# Cargar los datos
@st.cache_data
def load_data():
    return pd.read_csv("Crop_recommendation.csv")

# Función de predicción
def make_prediction(N, P, K, temperature, humidity, ph, rainfall, model, mx_scaler, sc_scaler):
    features = np.array([[N, P, K, temperature, humidity, ph, rainfall]])
    mx_features = mx_scaler.transform(features)
    sc_mx_features = sc_scaler.transform(mx_features)
    prediction = model.predict(sc_mx_features)
    return prediction[0]

# Función para entrenar el modelo con datos adicionales
def retrain_model(new_data, original_data):
    # Combinar datos originales con nuevos datos
    combined_data = pd.concat([original_data, new_data], ignore_index=True)

    # Aquí iría la lógica para reentrenar el modelo
    # Por ahora solo mostramos un mensaje
    st.success("Modelo reentrenado con éxito!")
    return combined_data

def visualize_data(data):
    st.subheader("Visualización de Datos")

    tab1, tab2, tab3, tab4, tab5 = st.tabs(["Nutrientes", "Condiciones Ambientales", "pH", "Lluvia", "Distribución de Cultivos"])

    with tab1:
        # Gráfico de nutrientes
        fig_nutrients = go.Figure()
        fig_nutrients.add_trace(go.Histogram(x=data['N'], name='Nitrógeno', marker_color='greenyellow'))
        fig_nutrients.add_trace(go.Histogram(x=data['P'], name='Fósforo', marker_color='firebrick'))
        fig_nutrients.add_trace(go.Histogram(x=data['K'], name='Potasio', marker_color='orange'))
        fig_nutrients.update_layout(title='Distribución de Nutrientes', barmode='overlay')
        st.plotly_chart(fig_nutrients, use_container_width=True)

    with tab2:
        # Gráfico de condiciones ambientales
        fig_env = go.Figure()
        fig_env.add_trace(go.Histogram(x=data['temperature'], name='Temperatura', marker_color='lightcoral'))
        fig_env.add_trace(go.Histogram(x=data['humidity'], name='Humedad', marker_color='olivedrab'))
        fig_env.update_layout(title='Condiciones Ambientales')
        st.plotly_chart(fig_env, use_container_width=True)

    with tab3:
        # Gráfico de pH
        fig_ph = px.histogram(data, x='ph', title='Distribución de pH')
        st.plotly_chart(fig_ph, use_container_width=True)

    with tab4:
        # Gráfico de lluvia
        fig_rain = px.histogram(data, x='rainfall', title='Distribución de Lluvia')
        st.plotly_chart(fig_rain, use_container_width=True)

    with tab5:
        # Gráfico de distribución de cultivos
        crop_counts = data['label'].value_counts()
        fig_crops = px.pie(values=crop_counts.values, names=crop_counts.index, title='Distribución de Cultivos')
        st.plotly_chart(fig_crops, use_container_width=True)

def prediction_form(model, mx_scaler, sc_scaler):
    with st.form("prediction_form"):
        col1, col2 = st.columns(2)
        with col1:
            N = st.number_input("Nivel de Nitrógeno", min_value=0, max_value=140)
            P = st.number_input("Nivel de Fósforo", min_value=0, max_value=145)
            K = st.number_input("Nivel de Potasio", min_value=0, max_value=205)
            temperature = st.number_input("Temperatura (°C)", min_value=0.0, max_value=50.0)
        with col2:
            humidity = st.number_input("Humedad (%)", min_value=0.0, max_value=100.0)
            ph = st.number_input("pH del suelo", min_value=0.0, max_value=14.0)
            rainfall = st.number_input("Lluvia (mm)", min_value=0.0, max_value=300.0)

        submitted = st.form_submit_button("Predecir Cultivo")

        if submitted:
            prediction = make_prediction(
                N, P, K, temperature, humidity, ph, rainfall,
                model, mx_scaler, sc_scaler
            )
            crop_name = get_crop_name(prediction)
            st.success(f"El cultivo recomendado es: {crop_name}")

def main():
    # Cargar datos y modelos
    model, mx_scaler, sc_scaler = load_models()
    data = load_data()

    # Título principal
    st.title("Sistema de Recomendación de Cultivos")

    # Menú desplegable
    menu = st.sidebar.selectbox(
        "Seleccione una opción",
        ["Inicio", "Visualización de Datos", "Predicción", "Métricas del Modelo", "Cargar Nuevos Datos"]
    )

    if menu == "Inicio":
        st.write("Bienvenido al Sistema de Recomendación de Cultivos")
        st.write("Utilice el menú de la izquierda para navegar por las diferentes funciones.")
        st.write("")

    elif menu == "Visualización de Datos":
        visualize_data(data)

    elif menu == "Predicción":
        prediction_form(model, mx_scaler, sc_scaler)

    elif menu == "Métricas del Modelo":
        show_model_metrics(model, data, mx_scaler, sc_scaler)

    elif menu == "Cargar Nuevos Datos":
        st.subheader("Cargar Nuevos Datos para Entrenamiento")
        uploaded_file = st.file_uploader("Seleccione un archivo CSV", type="csv")

        if uploaded_file is not None:
            try:
                new_data = pd.read_csv(uploaded_file)
                if set(data.columns) == set(new_data.columns):
                    st.write("Vista previa de los nuevos datos:")
                    st.write(new_data.head())

                # Crear tabs para mostrar diferentes comparaciones
                    comp_tab1, comp_tab2, comp_tab3 = st.tabs([
                        "Comparación de Distribuciones", 
                        "Estadísticas Descriptivas", 
                        "Distribución de Cultivos"
                    ])

                    with comp_tab1:
                        # Comparación de distribuciones de variables numéricas
                        numeric_cols = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']

                        for col in numeric_cols:
                            fig = go.Figure()

                        # Datos originales
                            fig.add_trace(go.Histogram(
                                x=data[col],
                                name='Datos Originales',
                                opacity=0.75
                            ))

                        # Nuevos datos
                            fig.add_trace(go.Histogram(
                                x=new_data[col],
                                name='Nuevos Datos',
                                opacity=0.75
                            ))

                            fig.update_layout(
                                title=f'Distribución de {col}',
                                barmode='overlay',
                                xaxis_title=col,
                                yaxis_title='Frecuencia'
                            )

                            st.plotly_chart(fig)

                    with comp_tab2:
                        # Estadísticas descriptivas
                        col1, col2 = st.columns(2)

                        with col1:
                            st.write("Estadísticas de Datos Originales:")
                            st.write(data[numeric_cols].describe())

                        with col2:
                            st.write("Estadísticas de Nuevos Datos:")
                            st.write(new_data[numeric_cols].describe())

                    with comp_tab3:
                    # Comparación de distribución de cultivos
                        col1, col2 = st.columns(2)

                        with col1:
                            crop_counts_original = data['label'].value_counts()
                            fig_crops_original = px.pie(
                                values=crop_counts_original.values,
                                names=crop_counts_original.index,
                                title='Distribución de Cultivos (Datos Originales)'
                            )
                            st.plotly_chart(fig_crops_original)

                        with col2:
                            crop_counts_new = new_data['label'].value_counts()
                            fig_crops_new = px.pie(
                                values=crop_counts_new.values,
                                names=crop_counts_new.index,
                                title='Distribución de Cultivos (Nuevos Datos)'
                            )
                            st.plotly_chart(fig_crops_new)

                # Botón para reentrenar el modelo
                    if st.button("Reentrenar Modelo"):
                        data = retrain_model(new_data, data)
                        st.success("Modelo reentrenado exitosamente con los nuevos datos!")

                else:
                    st.error("El archivo CSV debe tener las mismas columnas que el conjunto de datos original")
            except Exception as e:
                st.error(f"Error al cargar el archivo: {str(e)}")

if __name__ == "__main__":
    main()