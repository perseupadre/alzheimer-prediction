# 📓 Notebook - Desenvolvimento ML para Predição de Alzheimer

Notebook Jupyter completo com desenvolvimento do modelo DecisionTreeClassifier (92.56% acurácia) para predição de risco de Alzheimer, implementando pipeline reproduzível no Google Colab.

## 🎯 Objetivo

Sistema de apoio à decisão clínica com foco em:

**🔍 Triagem Eficiente**: Classificação binária simplificada  
**🏥 Aplicação Prática**: Integração com interface web  
**📊 Base Científica**: 2.149 amostras validadas (Kaggle)  
**⚡ Interpretabilidade**: DecisionTree facilmente explicável  
**📈 Monitoramento**: Histórico para acompanhamento longitudinal

## 🏆 Resultados Alcançados

### DecisionTreeClassifier - Modelo Final
- **🎯 Acurácia**: 92.56% (muito superior ao mínimo de 70%)
- **📊 F1-Score**: 92.54% 
- **⚖️ Sensibilidade**: 88.8% (detecção de casos positivos)
- **🛡️ Especificidade**: 94.6% (baixa taxa falsos positivos)
- **🔍 Interpretabilidade**: Árvore facilmente explicável

### Dataset Kaggle
- **Fonte**: [alzheimers-disease-dataset](https://www.kaggle.com/datasets/rabieelkharoua/alzheimers-disease-dataset)
- **Tamanho**: 2.149 amostras, 35 colunas (34 features + target)
- **Qualidade**: Dataset limpo, sem valores ausentes
- **Tipo**: Classificação binária (0 = Sem Alzheimer - Risco Baixo, 1 = Com Alzheimer - Risco Considerável)

### Features Críticas (Top 5)
1. **MMSE** (26.2%) - Mini Exame Estado Mental
2. **ADL** (21.3%) - Atividades Vida Diária
3. **FunctionalAssessment** (19.6%) - Avaliação Funcional  
4. **Age** - Idade do paciente
5. **Cognitive Symptoms** - Sintomas cognitivos

## 🧠 Pipeline de Desenvolvimento

### 1. EDA e Pré-processamento
- Análise exploratória, correlações, tratamento outliers
- StandardScaler + OneHotEncoder em pipeline integrado
- Estratificação e validação cruzada k-fold

### 2. Modelagem e Seleção
**Algoritmos testados**: LogisticRegression, KNN, DecisionTree, NaiveBayes, SVM  
**Otimização**: GridSearchCV para hiperparâmetros  
**Validação**: Cross-validation + hold-out test (20%)  
**Reproducibilidade**: Seed fixo (random_state=42)

### 3. Métricas e Thresholds
**Requisitos mínimos** (todos superados):  
- Accuracy ≥ 70% ✅ (alcançado: 92.56%)
- Precision ≥ 70% ✅ 
- Recall ≥ 70% ✅
- F1-Score ≥ 70% ✅
- AUC-ROC ≥ 70% ✅

## 🚀 Execução no Google Colab

### Link Direto
[**MVP_2_ML_PUCRIO_FINAL_v2.ipynb**](https://colab.research.google.com/drive/1-Ao-A9NNUVXYET737x-paAA6xFEKzOIr?usp=sharing)

### Tempo de Execução
- **Setup + Download**: ~3 min
- **EDA + Preprocessing**: ~8 min  
- **Modelagem + Otimização**: ~12 min
- **Avaliação + Exportação**: ~5 min
- **Total**: ~30 min

### Artefatos Gerados
- `best_model_pipeline.joblib` - Pipeline completo
- `feature_columns.joblib` - Lista de features  
- `model_info.json` - Metadados e performance

## 🔬 Metodologia Científica

**Ambiente**: Google Colab com desenvolvimento reproduzível  
**Validação**: Cross-validation k-fold + hold-out test estratificado  
**Interpretabilidade**: Feature importance, correlações, casos limite  
**Baseline Clínica**: MMSE como padrão-ouro, ADL como indicador funcional

## �️ Tecnologias

**Core**: Python 3.8+, scikit-learn, pandas, numpy  
**Visualização**: matplotlib, seaborn  
**Dataset**: kagglehub para download automático  
**Serialização**: joblib para exportação do modelo

## ⚠️ Limitações

**Dataset Sintético**: Baseado em dados simulados, não pacientes reais  
**Generalizabilidade**: Pode variar em diferentes populações  
**Uso Responsável**: Apoio à decisão, não substituto diagnóstico médico
