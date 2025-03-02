from db_connection import get_engine

# Obtener la conexión desde db_connection.py
connection = get_engine()

if connection:
    cursor = connection.cursor()
    cursor.execute("SELECT * FROM mdw22")
    rows = cursor.fetchall()
    for row in rows:
        print(row)

    connection.close()  # Cerramos la conexión después de usarla
    print("🔒 Conexión cerrada.")
else:
    print("⚠ No se pudo establecer la conexión a la base de datos.")
