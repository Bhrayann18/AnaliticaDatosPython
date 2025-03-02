from sqlalchemy import create_engine

def get_engine():
    """Crea y retorna un motor de conexión de SQLAlchemy."""
    try:
        engine = create_engine(
            "mssql+pyodbc://sa:Cltech*123@192.168.1.12/MiddlewareHus?driver=ODBC+Driver+17+for+SQL+Server"
        )
        return engine
    except Exception as e:
        print(f"❌ Error al crear el motor de conexión: {e}")
        return None
