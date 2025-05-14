import streamlit as st
import joblib
import pandas as pd
import numpy as np

# Cargar modelos
modelo_victoria = joblib.load("modelo_victoria.sav")
modelo_desempeno = joblib.load("modelo_desempeno.sav")
modelo_lesiones = joblib.load("modelo_lesiones.sav")


st.title("🏈 Predicciones NFL - Proyecto de Irvin")
opcion = st.sidebar.selectbox("Selecciona un objetivo", [
    "1. Predecir resultado de partido",
    "2. Predecir desempeño de jugador",
    "3. Recomendación de alineación",
    "4. Predicción de lesiones"
])

# 1. Predecir resultado de partido
if opcion == "1. Predecir resultado de partido":
    st.header("1. Predicción de Resultado del Partido")
    turnovers = st.number_input("Turnovers", min_value=0)
    yardas = st.number_input("Yardas totales", min_value=0)
    local = st.selectbox("Localía", ["Local", "Visitante"])

    if st.button("Predecir Resultado"):
        local_flag = 1 if local == "Local" else 0
        datos = pd.DataFrame([[yardas, turnovers, local_flag]], columns=["yardas", "turnovers", "local"])
        pred = modelo_victoria.predict(datos)[0]
        resultado = "Ganará" if pred == 1 else "Perderá"
        st.success(f"Resultado: {resultado}")

# 2. Predecir desempeño de jugador
elif opcion == "2. Predecir desempeño de jugador":
    st.header("2. Desempeño de Jugador")
    jugador = st.text_input("Nombre del jugador")
    rival = st.text_input("Nombre del rival")
    posicion = st.selectbox("Posición", ["QB", "RB", "WR", "TE", "DEF"])

    if st.button("Predecir Desempeño"):
        # Placeholder para features reales
        datos = pd.DataFrame([[jugador, rival, posicion]], columns=["player", "opponent", "position"])
        # Este modelo requiere que uses los mismos features que en entrenamiento
        st.warning("(🔧 Adaptar variables reales para predicción del modelo)")
        # pred = modelo_desempeno.predict(datos)
        # st.info(f"Desempeño estimado: {pred[0]}")

# 3. Recomendación de alineación
elif opcion == "3. Recomendación de alineación":
    st.header("3. Recomendación de Alineación")
    equipo = st.text_input("Nombre del equipo")
    rival = st.text_input("Nombre del rival")

    if st.button("Recomendar Jugadores"):
        st.info("(🔧 Este modelo requiere función de recomendación personalizada con base en modelo_desempeno)")

# 4. Predicción de lesiones
elif opcion == "4. Predicción de lesiones":
    st.header("4. Predicción de Lesión")
    jugador = st.text_input("Nombre del jugador")
    posicion = st.selectbox("Posición", ["QB", "RB", "WR", "TE", "DEF"])
    jugadas = st.number_input("Jugadas jugadas acumuladas", min_value=0)
    cesped = st.selectbox("Tipo de césped", ["Artificial", "Natural"])

    if st.button("Predecir Lesión"):
        cesped_flag = 1 if cesped == "Artificial" else 0
        datos = pd.DataFrame([[jugadas, cesped_flag]], columns=["snaps", "turf"])
        pred = modelo_lesiones.predict(datos)[0]
        riesgo = "Alto" if pred == 1 else "Bajo"
        st.warning(f"Riesgo de lesión: {riesgo}")
