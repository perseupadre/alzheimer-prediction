import sqlite3
import os
import json
from datetime import datetime

DATABASE_FILE = os.path.join(os.path.dirname(__file__), 'instance', 'site.db')

def init_db():
    """Inicializa o banco de dados e cria a tabela de histórico de predições."""
    os.makedirs(os.path.dirname(DATABASE_FILE), exist_ok=True)
    conn = sqlite3.connect(DATABASE_FILE)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS predictions_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            input_data TEXT NOT NULL,
            prediction INTEGER NOT NULL,
            probability REAL NOT NULL
        )
    ''')
    conn.commit()
    conn.close()
    print(f"Banco de dados SQLite inicializado em: {DATABASE_FILE}")

def add_prediction_to_history(input_data: dict, prediction: int, probability: float):
    """Adiciona uma nova predição ao histórico."""
    conn = sqlite3.connect(DATABASE_FILE)
    cursor = conn.cursor()
    timestamp = datetime.now().isoformat()
    input_data_json = json.dumps(input_data) # Armazena os dados de entrada como JSON
    cursor.execute(
        "INSERT INTO predictions_history (timestamp, input_data, prediction, probability) VALUES (?, ?, ?, ?)",
        (timestamp, input_data_json, prediction, probability)
    )
    conn.commit()
    conn.close()

def get_prediction_history():
    """Recupera todo o histórico de predições."""
    conn = sqlite3.connect(DATABASE_FILE)
    cursor = conn.cursor()
    cursor.execute("SELECT id, timestamp, input_data, prediction, probability FROM predictions_history ORDER BY timestamp DESC")
    history = []
    for row in cursor.fetchall():
        record = {
            "id": row[0],
            "timestamp": row[1],
            "input_data": json.loads(row[2]), # Carrega o JSON de volta para dict
            "prediction": row[3],
            "probability": row[4]
        }
        history.append(record)
    conn.close()
    return history

if __name__ == '__main__':
    # Este bloco será executado apenas se você rodar database.py diretamente
    init_db()
    print("Teste: Banco de dados inicializado. Tabela 'predictions_history' criada (se não existia).")