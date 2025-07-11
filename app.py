import pandas as pd
import streamlit as st
from catboost import CatBoostRegressor

# --- 1. Cargar modelo entrenado ---
modelo = CatBoostRegressor()
modelo.load_model("modelo_final.cbm")

# --- 2. Cargar archivo de categorias ---
df_cat = pd.read_csv('CSV categorical.csv', delimiter=';')

# --- 3. Definir variables que el usuario debe seleccionar ---
variables_usuario = [
    "det_tipo_proj",
    "det_tipo_inv",
    "det_gp",
    "det_gi",
    "det_area",
    "det_perf_inv",
    "ai_nivel3",
    "clase_proj"
]

st.title("🔮 Estimativa de Duração por Subfase")
st.markdown("Preencha as opções do projeto para obter a previsão de duração de cada subfase.")

# --- 4. Inputs do usuário com selectbox ---
inputs_usuario = {}
for var in variables_usuario:
    opcoes = sorted(df_cat[var].dropna().unique())
    valor = st.selectbox(f"{var.replace('_', ' ').capitalize()}", opcoes)
    inputs_usuario[var] = valor

# --- 5. Gerar subfases únicas mantendo ordem original ---
df_subfases = df_cat[["fas_fase", "fas_subfase"]].dropna().drop_duplicates().reset_index(drop=False)
orden_original = df_subfases["index"]
df_subfases = df_subfases.drop(columns=["index"])

# --- 6. Adicionar inputs do usuário a todas as subfases ---
for col, val in inputs_usuario.items():
    df_subfases[col] = val

# --- 7. Preparar para predição ---
columnas_modelo = modelo.feature_names_
df_model = df_subfases[columnas_modelo]

# --- 8. Prever e arredondar ---
df_subfases["duracao_estimada_dias"] = modelo.predict(df_model).round().astype(int)
df_subfases["orden"] = orden_original

# --- 9. Ordenar e exibir resultados ---
df_resultado = df_subfases.sort_values(by="orden").reset_index(drop=True)
df_resultado = df_resultado[["fas_fase", "fas_subfase", "duracao_estimada_dias"]]

st.subheader("📋 Resultado da previsão")
st.dataframe(df_resultado, use_container_width=True)

# --- 10. Botão para download ---
csv = df_resultado.to_csv(index=False).encode('utf-8')
st.download_button(
    label="📥 Baixar resultados em CSV",
    data=csv,
    file_name="estimativas_subfases.csv",
    mime="text/csv"
)
