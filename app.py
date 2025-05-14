import streamlit as st
import joblib
import pandas as pd
import numpy as np
import streamlit as st
import joblib
import pandas as pd
import numpy as np

# Cargar modelos
modelo_victoria = joblib.load("modelo_victoria.sav")
modelo_desempeno = joblib.load("modelo_desempeno.sav")

# Cargar dataset base para recomendación de jugadores
df_base = pd.read_csv("dataset_base.csv")

from xgboost import XGBClassifier

modelo_lesiones = XGBClassifier()
modelo_lesiones.load_model("modelo_lesiones.json")

st.title("🏈 Predicciones NFL - Proyecto de Irvin")
opcion = st.sidebar.selectbox("Selecciona un objetivo", [
    "1. Predecir resultado de partido",
    "2. Predecir desempeño de jugador",
    "3. Recomendación de alineación",
    "4. Predicción de lesiones"
])

# Diccionario de codificación de equipos
equipo_a_codigo = {
    'ARI': 0, 'ATL': 1, 'BAL': 2, 'BUF': 3, 'CAR': 4, 'CHI': 5, 'CIN': 6, 'CLE': 7,
    'DAL': 8, 'DEN': 9, 'DET': 10, 'GB': 11, 'HOU': 12, 'IND': 13, 'JAX': 14, 'KC': 15,
    'LAC': 16, 'LAR': 17, 'LV': 18, 'MIA': 19, 'MIN': 20, 'NE': 21, 'NO': 22, 'NYG': 23,
    'NYJ': 24, 'PHI': 25, 'PIT': 26, 'SEA': 27, 'SF': 28, 'TB': 29, 'TEN': 30, 'WAS': 31
}

equipos_lista = list(equipo_a_codigo.keys())

def mejores_jugadores_contra(mi_equipo, equipo_rival, season, surface, lesion_occurred, df_base, modelo, columnas_originales):
    jugadores_equipo = df_base[df_base[f"equipo_{mi_equipo}"] == 1]["jugador"].unique()
    lista_resultados = []

    for jugador in jugadores_equipo:
        jugador_col = f"player_name_{jugador}"
        rival_col = f"equipo_rival_{equipo_rival}"
        surface_col = "surface_natural" if surface == "grass" else "surface_sintetica"

        x = pd.DataFrame(data=np.zeros((1, len(columnas_originales))), columns=columnas_originales)

        if jugador_col in x.columns:
            x.loc[0, jugador_col] = 1
            x.loc[0, rival_col] = 1
            x.loc[0, surface_col] = 1
            x.loc[0, "season"] = season
            x.loc[0, "lesion_occurred"] = lesion_occurred

            proba = modelo.predict_proba(x)[0][1]
            lista_resultados.append({"jugador": jugador, "probabilidad": proba})

    resultados = pd.DataFrame(lista_resultados)
    resultados = resultados.sort_values(by="probabilidad", ascending=False).reset_index(drop=True)
    return resultados


# 1. Predecir resultado de partido
if opcion == "1. Predecir resultado de partido":
    st.header("1. Predicción de Resultado del Partido")

    home_team = st.selectbox("Equipo local", equipos_lista)
    away_team = st.selectbox("Equipo visitante", equipos_lista)
    home_encoded = equipo_a_codigo[home_team]
    away_encoded = equipo_a_codigo[away_team]

    home_yds_per_play = st.number_input("Yardas por jugada (local)", min_value=0.0)
    away_yds_per_play = st.number_input("Yardas por jugada (visitante)", min_value=0.0)
    home_time_of_poss = st.number_input("Tiempo de posesión (local)", min_value=0.0)
    away_time_of_poss = st.number_input("Tiempo de posesión (visitante)", min_value=0.0)
    home_plays = st.number_input("Jugadas totales (local)", min_value=0)
    away_plays = st.number_input("Jugadas totales (visitante)", min_value=0)
    home_drives = st.number_input("Drives (local)", min_value=0)
    away_drives = st.number_input("Drives (visitante)", min_value=0)
    home_yds_per_rush = st.number_input("Yardas por acarreo (local)", min_value=0.0)
    away_yds_per_rush = st.number_input("Yardas por acarreo (visitante)", min_value=0.0)
    home_yds_per_pass = st.number_input("Yardas por pase (local)", min_value=0.0)
    away_yds_per_pass = st.number_input("Yardas por pase (visitante)", min_value=0.0)
    home_fumbles = st.number_input("Fumbles (local)", min_value=0)
    away_fumbles = st.number_input("Fumbles (visitante)", min_value=0)
    home_ints = st.number_input("Intercepciones (local)", min_value=0)
    away_ints = st.number_input("Intercepciones (visitante)", min_value=0)

    if st.button("Predecir Resultado"):
        datos = pd.DataFrame([[
            home_encoded, away_encoded, home_yds_per_play, away_yds_per_play,
            home_time_of_poss, away_time_of_poss, home_plays, away_plays,
            home_drives, away_drives, home_yds_per_rush, away_yds_per_rush,
            home_yds_per_pass, away_yds_per_pass, home_fumbles, away_fumbles,
            home_ints, away_ints
        ]], columns=[
            'home_encoded', 'away_encoded', 'home_yds_per_play', 'away_yds_per_play',
            'home_time_of_poss', 'away_time_of_poss', 'home_plays', 'away_plays',
            'home_drives', 'away_drives', 'home_yds_per_rush', 'away_yds_per_rush',
            'home_yds_per_pass', 'away_yds_per_pass', 'home_fumbles', 'away_fumbles',
            'home_ints', 'away_ints'
        ])
        pred = modelo_victoria.predict(datos)[0]
        resultado = "Ganará" if pred == 1 else "Perderá"
        st.success(f"Resultado: {resultado}")

# 2. Predicción de desempeño de jugador (placeholder)
# 2. Predicción de desempeño de jugador
elif opcion == "2. Predecir desempeño de jugador":
    st.header("2. Desempeño de Jugador")

    jugador = st.text_input("Nombre exacto del jugador (ej. P.Garcon)")
    rival = st.selectbox("Equipo rival", [
        'ATL', 'BAL', 'BUF', 'CAR', 'CHI', 'CIN', 'CLE', 'DAL', 'DEN', 'DET', 'GB',
        'HOU', 'IND', 'JAX', 'KC', 'LAC', 'LAR', 'LV', 'MIA', 'MIN', 'NE', 'NO',
        'NYG', 'NYJ', 'PHI', 'PIT', 'SEA', 'SF', 'TB', 'TEN', 'WAS'
    ])
    temporada = st.number_input("Temporada", min_value=2016, max_value=2025, value=2021)
    cesped = st.selectbox("Tipo de césped", ["Natural", "Sintético"])
    lesion_ocurrida = st.selectbox("¿Tuvo una lesión esta temporada?", ["No", "Sí"])

    if st.button("Predecir Desempeño"):
        jugador_col = f"player_name_{jugador}"
        rival_col = f"equipo_rival_{rival}"
        surface_col = "surface_natural" if cesped == "Natural" else "surface_sintetica"

        # Crear DataFrame con ceros
        x = pd.DataFrame(data=np.zeros((1, len(modelo_desempeno.feature_names_in_))), columns=modelo_desempeno.feature_names_in_)

        # Asignar valores
        x.loc[0, jugador_col] = 1
        x.loc[0, rival_col] = 1
        x.loc[0, surface_col] = 1
        x.loc[0, "season"] = temporada
        x.loc[0, "lesion_occurred"] = 1 if lesion_ocurrida == "Sí" else 0

        # Predicción
        proba = modelo_desempeno.predict_proba(x)[0][1]
        clase = modelo_desempeno.predict(x)[0]

        st.markdown(f"🎯 **Predicción para {jugador} vs {rival}:**")
        st.write(f"Probabilidad de alto desempeño: {proba:.2f} → Clasificación: {clase}")


# 3. Recomendación de alineación (placeholder)
# 3. Recomendación de alineación
elif opcion == "3. Recomendación de alineación":
    st.header("3. Recomendación de Alineación")

    mi_equipo = st.selectbox("Tu equipo", [
        'ATL', 'BAL', 'BUF', 'CAR', 'CHI', 'CIN', 'CLE', 'DAL', 'DEN', 'DET', 'GB',
        'HOU', 'IND', 'JAX', 'KC', 'LAC', 'LAR', 'LV', 'MIA', 'MIN', 'NE', 'NO',
        'NYG', 'NYJ', 'PHI', 'PIT', 'SEA', 'SF', 'TB', 'TEN', 'WAS'
    ])
    rival = st.selectbox("Equipo rival", [
        'ATL', 'BAL', 'BUF', 'CAR', 'CHI', 'CIN', 'CLE', 'DAL', 'DEN', 'DET', 'GB',
        'HOU', 'IND', 'JAX', 'KC', 'LAC', 'LAR', 'LV', 'MIA', 'MIN', 'NE', 'NO',
        'NYG', 'NYJ', 'PHI', 'PIT', 'SEA', 'SF', 'TB', 'TEN', 'WAS'
    ])
    temporada = st.number_input("Temporada", min_value=2016, max_value=2025, value=2021)
    cesped = st.selectbox("Tipo de césped", ["Natural", "Sintético"])
    lesion_ocurrida = st.selectbox("¿Considerar jugadores lesionados?", ["Sí", "No"])

    if st.button("Recomendar Jugadores"):
        incluir_lesionados = 1 if lesion_ocurrida == "Sí" else 0

        try:
            resultados = mejores_jugadores_contra(
                mi_equipo=mi_equipo,
                equipo_rival=rival,
                season=temporada,
                surface="grass" if cesped == "Natural" else "turf",
                lesion_occurred=incluir_lesionados,
                df_base=df_base,
                modelo=modelo_desempeno,
                columnas_originales=modelo_desempeno.feature_names_in_
            )

            top3 = resultados.head(3)

            st.markdown(f"🎯 **Top 3 jugadores de {mi_equipo} contra {rival}:**")
            for _, row in top3.iterrows():
                st.write(f"  {row['jugador']:<20} → Probabilidad alto desempeño: {row['probabilidad']:.2f}")

        except Exception as e:
            st.error(f"⚠️ Error al generar recomendaciones: {e}")


# 4. Predicción de lesiones
elif opcion == "4. Predicción de lesiones":
    st.header("4. Predicción de Lesión")

    equipo = st.selectbox("Equipo del jugador", [
        'ATL', 'BAL', 'BUF', 'CAR', 'CHI', 'CIN', 'CLE', 'DAL', 'DEN', 'DET', 'GB',
        'HOU', 'IND', 'JAX', 'KC', 'LA', 'LAC', 'MIA', 'MIN', 'NE', 'NO', 'NYG', 'NYJ',
        'OAK', 'PHI', 'PIT', 'SEA', 'SF', 'TB', 'TEN', 'WAS'
    ])
    superficie = st.selectbox("Tipo de césped", ["Natural", "Sintética"])

    plays_before = st.number_input("Jugadas antes de la lesión", min_value=0)
    pass_rush = st.number_input("Proporción pase/acarreos (0 a 1)", min_value=0.0, max_value=1.0, value=0.5)
    load = st.number_input("Carga de rendimiento (performance load)", min_value=0.0)
    plays_season = st.number_input("Jugadas en la temporada", min_value=0)
    prev_injury = st.selectbox("¿Lesión previa?", ["No", "Sí"])

    if st.button("Predecir Lesión"):
        columnas = modelo_lesiones.feature_names_in_
        datos = pd.DataFrame(data=[np.zeros(len(columnas))], columns=columnas)

        # Asignar valores
        datos.at[0, 'plays_before_injury'] = plays_before
        datos.at[0, 'pass_rush_ratio'] = pass_rush
        datos.at[0, 'performance_load'] = load
        datos.at[0, 'plays_per_season'] = plays_season
        datos.at[0, 'prev_injury'] = 1 if prev_injury == "Sí" else 0

        equipo_col = f"equipo_{equipo}"
        superficie_col = "surface_natural" if superficie == "Natural" else "surface_sintetica"

        if equipo_col in datos.columns:
            datos.at[0, equipo_col] = 1
        datos.at[0, superficie_col] = 1

        pred = modelo_lesiones.predict(datos)[0]
        riesgo = "Alto" if pred == 1 else "Bajo"
        st.warning(f"Riesgo de lesión: {riesgo}")

