from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import joblib
import pandas as pd
import os
from flasgger import Swagger # Para documentação da API
from database import init_db, add_prediction_to_history, get_prediction_history

# --- Configurações da Aplicação ---
app = Flask(__name__, static_folder='../frontend', static_url_path='/')
CORS(app) # Habilita CORS para permitir requisições do frontend
swagger = Swagger(app) # Inicializa o Swagger

# Caminhos para o modelo e as colunas de features
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'trained_model', 'best_model_pipeline.joblib')
FEATURE_COLUMNS_PATH = os.path.join(os.path.dirname(__file__), 'trained_model', 'feature_columns.joblib')

# Carrega o pipeline e as colunas de features ao iniciar a aplicação
ml_pipeline = None
feature_columns = None

try:
    ml_pipeline = joblib.load(MODEL_PATH)
    feature_columns = joblib.load(FEATURE_COLUMNS_PATH)
    print("Modelo e colunas de features carregados com sucesso!")
    print(f"Colunas esperadas pelo modelo: {feature_columns}")
except Exception as e:
    print(f"Erro ao carregar o modelo ou as colunas de features: {e}")
    print("Certifique-se de que os arquivos 'best_model_pipeline.joblib' e 'feature_columns.joblib' estão na pasta 'backend/trained_model'.")

# Inicializa o banco de dados
init_db()

# --- Rotas da API ---

@app.route('/')
def index():
    """
    Rota principal que serve o arquivo index.html do frontend.
    """
    return send_from_directory(app.static_folder, 'index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """
    Endpoint para predição de Alzheimer.
    ---
    parameters:
      - name: body
        in: body
        required: true
        schema:
          type: object
          properties:
            Age: {type: number, description: 'Idade do paciente'}
            Gender: {type: string, description: 'Gênero (0=Masculino, 1=Feminino)'}
            Ethnicity: {type: string, description: 'Etnia (Ex: Branco, Negro, Asiático, Outro)'}
            EducationLevel: {type: string, description: 'Nível de educação'}
            BMI: {type: number, description: 'Índice de Massa Corporal'}
            Smoking: {type: number, description: 'Fuma? (0=Não, 1=Sim)'}
            AlcoholConsumption: {type: number, description: 'Consumo de Álcool (unidades por semana)'}
            PhysicalActivity: {type: number, description: 'Atividade Física (horas por semana)'}
            DietQuality: {type: number, description: 'Qualidade da Dieta (escala 0-10)'}
            SleepQuality: {type: number, description: 'Qualidade do Sono (escala 0-10)'}
            FamilyHistoryAlzheimers: {type: number, description: 'Histórico Familiar de Alzheimer (0=Não, 1=Sim)'}
            CardiovascularDisease: {type: number, description: 'Doença Cardiovascular (0=Não, 1=Sim)'}
            Diabetes: {type: number, description: 'Diabetes (0=Não, 1=Sim)'}
            Depression: {type: number, description: 'Depressão (0=Não, 1=Sim)'}
            HeadInjury: {type: number, description: 'Lesão na Cabeça (0=Não, 1=Sim)'}
            Hypertension: {type: number, description: 'Hipertensão (0=Não, 1=Sim)'}
            SystolicBP: {type: number, description: 'Pressão Arterial Sistólica (mmHg)'}
            DiastolicBP: {type: number, description: 'Pressão Arterial Diastólica (mmHg)'}
            CholesterolTotal: {type: number, description: 'Colesterol Total (mg/dL)'}
            CholesterolLDL: {type: number, description: 'Colesterol LDL (mg/dL)'}
            CholesterolHDL: {type: number, description: 'Colesterol HDL (mg/dL)'}
            CholesterolTriglycerides: {type: number, description: 'Triglicerídeos (mg/dL)'}
            MMSE: {type: number, description: 'Mini Exame do Estado Mental (escala 0-30)'}
            FunctionalAssessment: {type: number, description: 'Avaliação Funcional (escala 0-10)'}
            MemoryComplaints: {type: number, description: 'Queixas de Memória (0=Não, 1=Sim)'}
            BehavioralProblems: {type: number, description: 'Problemas Comportamentais (0=Não, 1=Sim)'}
            ADL: {type: number, description: 'Atividades da Vida Diária (escala 0-10)'}
            Confusion: {type: number, description: 'Confusão Mental (0=Não, 1=Sim)'}
            Disorientation: {type: number, description: 'Desorientação (0=Não, 1=Sim)'}
            PersonalityChanges: {type: number, description: 'Mudanças de Personalidade (0=Não, 1=Sim)'}
            DifficultyCompletingTasks: {type: number, description: 'Dificuldade em Completar Tarefas (0=Não, 1=Sim)'}
            Forgetfulness: {type: number, description: 'Esquecimento (0=Não, 1=Sim)'}
            DoctorInCharge: {type: string, description: 'Médico Responsável'}
    responses:
      200:
        description: 'Resultado da predição.'
        schema:
          type: object
          properties:
            prediction: {type: integer, description: '0 para Baixo Risco, 1 para Risco Considerável'}
            probability: {type: number, description: 'Probabilidade da classe 1 (Risco Considerável)'}
            message: {type: string, description: 'Mensagem descritiva do resultado'}
      400:
        description: 'Erro nos dados de entrada.'
      500:
        description: 'Erro interno do servidor.'
    """
    if ml_pipeline is None or feature_columns is None:
        return jsonify({"error": "Modelo não carregado. Verifique os logs do servidor."}), 500

    data = request.get_json()
    if not data:
        return jsonify({"error": "Dados JSON não fornecidos."}), 400

    # Cria um DataFrame com os dados de entrada, garantindo a ordem e preenchendo faltantes
    # com NaN para que o pipeline de pré-processamento possa lidar com eles
    # É CRUCIAL QUE 'data' CONTENHA TODAS AS COLUNAS EM 'feature_columns'
    input_df = pd.DataFrame([data], columns=feature_columns)

    # Verifica se todas as colunas esperadas estão presentes
    # Este check é fundamental e foi o motivo do seu erro anterior
    missing_cols = set(feature_columns) - set(input_df.columns)
    if missing_cols:
        return jsonify({"error": f"Colunas ausentes nos dados de entrada: {', '.join(missing_cols)}"}), 400

    try:
        prediction = ml_pipeline.predict(input_df)[0]
        probability = ml_pipeline.predict_proba(input_df)[:, 1][0] # Probabilidade da classe 1 (Alzheimer)

        # Adiciona a predição ao histórico
        add_prediction_to_history(data, int(prediction), float(probability))

        result = {
            "prediction": int(prediction),
            "probability": float(probability),
            "message": "Predisposição para Alzheimer: Sim - Risco Considerável" if prediction == 1 else "Predisposição para Alzheimer: Não - Risco Baixo"
        }
        return jsonify(result)
    except Exception as e:
        print(f"Erro na predição: {e}")
        return jsonify({"error": f"Erro ao processar a predição: {e}"}), 500

@app.route('/history', methods=['GET'])
def history():
    """
    Endpoint para recuperar o histórico de predições.
    ---
    responses:
      200:
        description: 'Lista de predições anteriores.'
        schema:
          type: array
          items:
            type: object
            properties:
              id: {type: integer}
              timestamp: {type: string}
              input_data: {type: object}
              prediction: {type: integer}
              probability: {type: number}
    """
    try:
        history_data = get_prediction_history()
        return jsonify(history_data)
    except Exception as e:
        print(f"Erro ao recuperar histórico: {e}")
        return jsonify({"error": f"Erro ao recuperar o histórico de predições: {e}"}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)