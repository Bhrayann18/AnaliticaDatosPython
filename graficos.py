import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from db_connection import get_engine
from statsmodels.tsa.arima.model import ARIMA
from sqlalchemy import create_engine
from statsmodels.tsa.arima.model import ARIMA
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.cluster import KMeans
from sqlalchemy import create_engine

# 🔹 Cargar datos de pacientes procesados por conexión
def load_datos_por_conexion():
    """Carga los datos de exámenes, órdenes y pacientes procesados por cada equipo."""
    engine = get_engine()
    if engine is None:
        print("❌ No se pudo obtener la conexión a la base de datos.")
        return None

    query = """
        SELECT 
            mdw10.mdw10c1 AS conexion_id, 
            mdw10.mdw10c2 AS conexion_nombre, 
            mdw23.mdw23c17 AS fecha_examen,
            mdw22.mdw22c1 AS orden_id,
            mdw21.mdw21c1 AS paciente_id
        FROM mdw23
        JOIN mdw22 ON mdw23.mdw22c1 = mdw22.mdw22c1
        JOIN mdw21 ON mdw22.mdw21c1 = mdw21.mdw21c1
        JOIN mdw17 ON mdw23.mdw17c1 = mdw17.mdw17c1
        JOIN mdw30 ON mdw17.mdw17c1 = mdw30.mdw17c1
        JOIN mdw10 ON mdw30.mdw10c1 = mdw10.mdw10c1
    """

    try:
        with engine.connect() as connection:
            df = pd.read_sql(query, connection)

        # Convertir fecha_examen a tipo datetime
        df["fecha_examen"] = pd.to_datetime(df["fecha_examen"])
        df["fecha"] = df["fecha_examen"].dt.date  # Extraer solo la fecha
        df["dia_semana"] = df["fecha_examen"].dt.weekday  # Extraer día de la semana
        df["dia_mes"] = df["fecha_examen"].dt.day  # Extraer día del mes

        # Calcular la media de pacientes procesados en los últimos 7 días
        df["pacientes_7d"] = df.groupby("conexion_nombre")["paciente_id"].transform(lambda x: x.rolling(7, min_periods=1).count())

        return df
    except Exception as e:
        print(f"❌ Error al cargar los datos: {e}")
        return None
# Diagrama de barras (Pacientes Procesados)
def plot_bar_chart(df):
    """Genera un gráfico de barras mostrando la cantidad de pacientes procesados por conexión agregados cada 30 días."""
    if df is None or df.empty:
        print("⚠ No hay datos disponibles para mostrar.")
        return

    # Convertir fecha a tipo datetime si no lo es
    df["fecha"] = pd.to_datetime(df["fecha"])

    # Crear una columna de período mensual
    df["mes"] = df["fecha"].dt.to_period("M")

    # Agrupar por mes y conexión para obtener la cantidad de pacientes procesados
    registros = df.groupby(["mes", "conexion_nombre"]).agg(
        total_pacientes=("paciente_id", "nunique")
    ).reset_index()

    # Convertir a formato de fecha para graficar correctamente
    registros["mes"] = registros["mes"].astype(str)

    plt.figure(figsize=(14, 7))
    sns.barplot(x="mes", y="total_pacientes", hue="conexion_nombre", data=registros, palette="viridis")
    plt.title("Pacientes Atendidos por Conexión Agregados Cada 30 Días")
    plt.xlabel("Mes")
    plt.ylabel("Cantidad de Pacientes")
    plt.xticks(rotation=45)
    plt.legend(title="Conexión", bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()
    plt.show()

# 🔹 Histograma por Conexión (Pacientes Procesados)
def plot_histogram(df):
    """Genera histogramas para analizar la distribución de pacientes por conexión."""
    if df is None or df.empty:
        print("⚠ No hay datos disponibles para mostrar.")
        return

    plt.figure(figsize=(12, 6))
    sns.histplot(data=df, x="fecha", hue="conexion_nombre", multiple="stack", bins=30, kde=True)
    plt.title("Distribución de Pacientes por Conexión")
    plt.xlabel("Fecha")
    plt.ylabel("Frecuencia")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

# 🔹 Mapa de Calor por Conexión (Pacientes Procesados)
def plot_heatmap(df):
    """Genera un mapa de calor con la cantidad de pacientes por conexión con formato de fecha mejorado."""
    if df is None or df.empty:
        print("⚠ No hay datos disponibles para mostrar.")
        return
    
    # Convertir fecha a tipo datetime y formatearla correctamente
    df["fecha"] = pd.to_datetime(df["fecha"]).dt.strftime("%Y-%m-%d")

    # Agrupar por fecha y conexión para contar pacientes únicos
    registros = df.groupby(["fecha", "conexion_nombre"]).agg(
        total_pacientes=("paciente_id", "nunique")
    ).reset_index()

    # Crear tabla pivote con valores agregados
    pivot_df = registros.pivot_table(values="total_pacientes", index="fecha", columns="conexion_nombre", aggfunc="sum").fillna(0)

    plt.figure(figsize=(14, 8))
    ax = sns.heatmap(pivot_df, cmap="coolwarm", linewidths=0.5)

    plt.title("Mapa de Calor: Pacientes Procesados por Conexión en el Tiempo")
    plt.xlabel("Conexión")
    plt.ylabel("Fecha")

    # Ajustar formato de las etiquetas del eje Y para mostrar solo Año-Mes-Día
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=10)

    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

# 🔹 Modelo Predictivo con ARIMA (Fase 4)
def modelar_arima(df):
    """Entrena un modelo ARIMA para predecir la cantidad de pacientes procesados por conexión."""
    if df is None or df.empty:
        print("⚠ No hay datos disponibles para entrenar el modelo.")
        return

    # Convertir la fecha a formato datetime correcto
    df["fecha"] = pd.to_datetime(df["fecha"])
    
    # Agrupar por fecha y conexión, contar pacientes únicos procesados
    df_pred = df.groupby(["fecha", "conexion_nombre"]).agg(
        total_pacientes=("paciente_id", "nunique")
    ).reset_index()

    # Obtener lista de conexiones únicas
    conexiones = df_pred["conexion_nombre"].unique()

    # Diccionario para almacenar predicciones por conexión
    predicciones_por_conexion = {}

    plt.figure(figsize=(12, 6))

    for conexion in conexiones:
        df_conexion = df_pred[df_pred["conexion_nombre"] == conexion].copy()
        df_conexion.set_index("fecha", inplace=True)
        df_conexion.index = pd.to_datetime(df_conexion.index)  # Asegurar que es datetime
        df_conexion = df_conexion.asfreq("D")  # Configurar frecuencia diaria
        serie = df_conexion["total_pacientes"].fillna(method="ffill")  # Llenar valores faltantes

        if len(serie) < 10:
            print(f"⚠ No hay suficientes datos para modelar ARIMA en {conexion}.")
            continue

        # Separar en datos de entrenamiento y prueba
        train_size = int(len(serie) * 0.8)
        train, test = serie[:train_size], serie[train_size:]

        # Ajustar el modelo ARIMA
        model = ARIMA(train, order=(5, 1, 0))  # (p=5, d=1, q=0) -> Ajustable
        model_fit = model.fit()

        # Generar predicciones para los próximos 30 días
        forecast_steps = len(test) + 30
        forecast = model_fit.forecast(steps=forecast_steps)

        # Crear DataFrame con predicciones
        forecast_dates = pd.date_range(start=test.index[0], periods=forecast_steps, freq="D")
        df_forecast = pd.DataFrame({"fecha": forecast_dates, "prediccion": forecast})
        predicciones_por_conexion[conexion] = df_forecast

        # Graficar predicción
        plt.plot(serie, label=f"Datos Reales - {conexion}", linestyle="dotted")
        plt.plot(df_forecast["fecha"], df_forecast["prediccion"], label=f"Predicción - {conexion}")

    plt.axvline(x=test.index[0], linestyle="--", color="gray", label="Inicio de Predicción")
    plt.title("Predicción de Pacientes Procesados por Conexión (ARIMA)")
    plt.xlabel("Fecha")
    plt.ylabel("Cantidad de Pacientes")
    plt.legend()
    plt.grid(True)
    plt.show()

    print("\nPredicciones para los próximos 30 días por conexión:")
    for conexion, pred_df in predicciones_por_conexion.items():
        print(f"\nConexión: {conexion}")
        print(pred_df.tail(30))

# 🔹 Modelo de Regresión y Random Forest por Conexión con Fecha como Variable Predictora
def modelar_regresion_random_forest_por_conexion(df):
    """Aplica Regresión Lineal y Random Forest optimizado para predecir la cantidad de pacientes procesados por conexión considerando la fecha."""
    if df is None or df.empty:
        print("⚠ No hay datos disponibles para entrenar el modelo.")
        return
    
    # Convertir fecha a número secuencial (días desde la primera fecha)
    df["fecha"] = pd.to_datetime(df["fecha"])
    df["dias_desde_inicio"] = (df["fecha"] - df["fecha"].min()).dt.days
    
    conexiones = df["conexion_nombre"].unique()
    resultados = []
    
    for conexion in conexiones:
        print(f"\n📡 Entrenando modelo para la conexión: {conexion}")
        df_conexion = df[df["conexion_nombre"] == conexion]

        df_pred = df_conexion.groupby(["fecha", "dias_desde_inicio", "dia_semana", "dia_mes"]).agg(
            total_pacientes=("paciente_id", "nunique"),
            pacientes_7d=("pacientes_7d", "mean")
        ).reset_index()

        # Definir variables predictoras y objetivo
        X = df_pred[["dias_desde_inicio", "dia_semana", "dia_mes", "pacientes_7d"]].fillna(0)
        y = df_pred["total_pacientes"]

        # Dividir en entrenamiento y prueba
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Modelo de Regresión Lineal
        lr_model = LinearRegression()
        lr_model.fit(X_train, y_train)
        y_pred_lr = lr_model.predict(X_test)
        
        # Optimización de Random Forest con GridSearchCV
        param_grid = {"n_estimators": [50, 100, 200], "max_depth": [None, 10, 20]}
        rf_grid = GridSearchCV(RandomForestRegressor(random_state=42), param_grid, cv=3, scoring='r2')
        rf_grid.fit(X_train, y_train)
        best_rf_model = rf_grid.best_estimator_
        y_pred_rf = best_rf_model.predict(X_test)
        
        # Evaluación de modelos
        resultado = {
            "conexion": conexion,
            "mae_lr": mean_absolute_error(y_test, y_pred_lr),
            "rmse_lr": np.sqrt(mean_squared_error(y_test, y_pred_lr)),
            "r2_lr": r2_score(y_test, y_pred_lr),
            "mejores_parametros_rf": rf_grid.best_params_,
            "mae_rf": mean_absolute_error(y_test, y_pred_rf),
            "rmse_rf": np.sqrt(mean_squared_error(y_test, y_pred_rf)),
            "r2_rf": r2_score(y_test, y_pred_rf)
        }
        resultados.append(resultado)

        print("\n🔹 Resultados de Regresión Lineal:")
        print(f"MAE: {resultado['mae_lr']}")
        print(f"RMSE: {resultado['rmse_lr']}")
        print(f"R²: {resultado['r2_lr']}")

        print("\n🔹 Resultados de Random Forest (Optimizado):")
        print(f"Mejores Parámetros: {resultado['mejores_parametros_rf']}")
        print(f"MAE: {resultado['mae_rf']}")
        print(f"RMSE: {resultado['rmse_rf']}")
        print(f"R²: {resultado['r2_rf']}")

        # 🔹 Visualización de Predicciones vs Datos Reales
        plt.figure(figsize=(10, 5))
        plt.plot(y_test.values, label="Valores Reales", color="blue")
        plt.plot(y_pred_rf, label="Predicción Random Forest", color="red", linestyle="dashed")
        plt.title(f"Comparación de Predicción vs Datos Reales ({conexion})")
        plt.xlabel("Índice de Prueba")
        plt.ylabel("Pacientes Procesados")
        plt.legend()
        plt.show()
    
    return pd.DataFrame(resultados)


# 🔹 Clustering de Conexiones
def clustering_conexiones(df):
    """Aplica K-Means para segmentar conexiones según la cantidad de pacientes procesados."""
    if df is None or df.empty:
        print("⚠ No hay datos disponibles para clustering.")
        return

    df_cluster = df.groupby("conexion_nombre").agg(
        total_pacientes=("paciente_id", "nunique")
    ).reset_index()

    # Aplicar K-Means con 3 clusters
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    df_cluster["cluster"] = kmeans.fit_predict(df_cluster[["total_pacientes"]])

    # Visualización del clustering
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df_cluster, x="conexion_nombre", y="total_pacientes", hue="cluster", palette="viridis", s=100)
    plt.title("Segmentación de Conexiones por Pacientes Procesados (K-Means)")
    plt.xlabel("Conexión")
    plt.ylabel("Cantidad de Pacientes")
    plt.xticks(rotation=45)
    plt.legend(title="Cluster")
    plt.show()

    print("\nSegmentación de conexiones completada. Clusters asignados:")
    print(df_cluster)


# 🔹 Ejecutar los análisis
if __name__ == "__main__":
    print("📡 Cargando datos de exámenes, órdenes y pacientes por conexión...")
    df_datos = load_datos_por_conexion()

    if df_datos is not None:
        print("📊 Generando gráfico de barras por conexión...")
        plot_bar_chart(df_datos)

        print("📊 Generando histogramas por conexión...")
        plot_histogram(df_datos)

        print("📊 Generando mapas de calor por conexión...")
        plot_heatmap(df_datos)

        print("📊 Ejecutando modelo predictivo ARIMA...")
        modelar_arima(df_datos)

        print("📊 Ejecutando modelo random forest..")
        modelar_regresion_random_forest_por_conexion(df_datos)

        print("📊 Ejecutando Clustering de Conexiones...")
        clustering_conexiones(df_datos)