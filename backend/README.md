# 🧠 Backend - API Flask para Predição de Alzheimer

API REST que serve o modelo DecisionTreeClassifier (92.56% acurácia) para predição de risco de Alzheimer com validação automática e documentação Swagger.

## 🚀 Quick Start

```bash
# Instalar dependências
pip install -r requirements.txt

# Executar servidor
python app.py

# Acessar sistema
# Interface: http://localhost:5000
# API Docs: http://localhost:5000/apidocs/
```

## � API Endpoints

### `POST /predict`
Realiza predição de risco de Alzheimer
- **Input**: JSON com features médicas
- **Output**: `{"prediction": 0|1, "probability": float, "message": string}`
- **Validação**: Automática para 32 features obrigatórias/opcionais

### `GET /history` 
Recupera histórico de predições anteriores
- **Output**: Array JSON com predições salvas

### `GET /`
Serve interface web do frontend automaticamente

### `GET /apidocs/`
Documentação Swagger interativa da API

## 📊 Features do Modelo (32 variáveis)

**Obrigatórias**: Age, Gender, Ethnicity, EducationLevel, Height, Weight, BMI  
**Opcionais**: Smoking, AlcoholConsumption, PhysicalActivity, DietQuality, SleepQuality, FamilyHistoryAlzheimers, CardiovascularDisease, Diabetes, Depression, HeadInjury, Hypertension, SystolicBP, DiastolicBP, CholesterolTotal, CholesterolLDL, CholesterolHDL, CholesterolTriglycerides, MMSE, FunctionalAssessment, ADL, MemoryComplaints, BehavioralProblems, Confusion, Disorientation, PersonalityChanges, DifficultyCompletingTasks, Forgetfulness

## 🧪 Testes Automatizados

```bash
# Executar testes de performance (refatorados com boas práticas)
python -m pytest tests/ -v

# Script de teste rápido (nova arquitetura orientada a funções)
cd .. && python run_tests.py
```

**Arquitetura de Testes**: Classe `TestModelPerformanceRequirements` organizada com métodos especializados  
**Validações**: Accuracy, Precision, Recall, F1-Score, AUC-ROC ≥ 70%  
**Performance**: Testes executam em ~3s com dados sintéticos otimizados

## 💾 Banco SQLite

**Tabela**: `predictions_history`  
**Campos**: id, timestamp, input_data (JSON), prediction, probability  
**Função**: Armazenar histórico de predições para acompanhamento

## ⚠️ Importante

**Uso Clínico**: Ferramenta de apoio à decisão, não substitui diagnóstico médico  
**Dados Obrigatórios**: Apenas campos demográficos básicos  
**Valores Padrão**: Campos opcionais usam valores clinicamente apropriados  
**Integração**: Frontend servido automaticamente na rota principal
