# 🧠 Sistema de Predição de Risco de Alzheimer

Sistema completo de Machine Learning para apoio à decisão clínica na predição de risco de Alzheimer, implementando solução end-to-end com alta performance (92.56% de acurácia).

## 🎯 Visão Geral

**DecisionTreeClassifier** integrado em sistema full-stack com:

- **🤖 Modelo ML**: 92.56% acurácia, pipeline scikit-learn otimizado
- **🌐 API REST**: Flask com Swagger, validação automática
- **💻 Interface Web**: Responsiva, 32 features médicas categorizadas  
- **🧪 Testes PyTest**: Thresholds ≥70%, validação contínua
- **📊 Notebook Colab**: Desenvolvimento reproduzível completo

## 📁 Estrutura do Projeto

```
PROJETO_02/
├── README.md                             # Documentação principal (este arquivo)
├── pyproject.toml                        # Configurações do projeto e dependências
├── run_tests.py                          # Script refatorado para testes de performance
│
├── backend/                              # 🔧 Servidor Flask e API
│   ├── README.md                         # Documentação do backend
│   ├── app.py                            # Aplicação Flask principal
│   ├── database.py                       # Gerenciamento do banco SQLite
│   ├── requirements.txt                  # Dependências Python
│   ├── instance/                                      
│   │   └── site.db                       # Banco SQLite (histórico de predições)
│   ├── trained_model/
│   │   ├── best_model_pipeline.joblib    # Pipeline do modelo treinado
│   │   ├── feature_columns.joblib        # Colunas de features
│   │   └── model_info.json               # Metadados do modelo
│   └── tests/                            # 🧪 Testes automatizados
│       ├── README.md                     # Documentação dos testes
│       ├── conftest.py                   # Configurações do pytest
│       └── test_model_performance.py     # Testes de performance
│
├── frontend/                             # 🌐 Interface Web
│   ├── README.md                         # Documentação do frontend
│   ├── index.html                        # Página principal
│   ├── script.js                         # Lógica JavaScript
│   └── style.css                         # Estilos CSS
│
└── notebook/                             # 📓 Desenvolvimento do Modelo
    ├── README.md                         # Documentação do notebook
    └── MVP_2_ML_PUCRIO_FINAL_v2.ipynb    # Notebook completo do desenvolvimento
```

## 🚀 Quick Start

### Instalação e Execução
```bash
# 1. Instalar dependências
pip install -r backend/requirements.txt

# 2. Executar servidor Flask
cd backend && python app.py

# 3. Acessar sistema
# Interface Web: http://localhost:5000
# API Docs: http://localhost:5000/apidocs/
```

### Testes
```bash
# Executar script de teste rápido (refatorado com boas práticas)
python run_tests.py

# Executar testes PyTest completos
python -m pytest backend/tests/ -v
```

## 🏆 Performance do Sistema

### Métricas do Modelo (DecisionTreeClassifier)
- **Acurácia**: 92.56% (meta: ≥70%)
- **F1-Score**: 92.54% 
- **Sensibilidade**: 88.8%
- **Especificidade**: 94.6%

### Features Críticas (Top 5)
1. **MMSE** (26.2%) - Mini Exame Estado Mental
2. **ADL** (21.3%) - Atividades Vida Diária  
3. **FunctionalAssessment** (19.6%) - Avaliação Funcional
4. **Age** - Idade do paciente
5. **Cognitive Symptoms** - Sintomas cognitivos

### Dataset
- **Fonte**: [Kaggle Alzheimer's Dataset](https://www.kaggle.com/datasets/rabieelkharoua/alzheimers-disease-dataset)
- **Amostras**: 2.149 casos validados
- **Features**: 32 variáveis médicas categorizadas

## 🛠️ Tecnologias

**Backend**: Python 3.8+, Flask 3.1.1, scikit-learn 1.6.1, SQLite, pytest  
**Frontend**: HTML5/CSS3, JavaScript ES6+, Design Responsivo  
**ML**: Jupyter Notebook, Google Colab, Kaggle Hub

## ⚠️ Considerações Importantes

**Uso Clínico**: Ferramenta de apoio à decisão, não substitui diagnóstico médico  
**Validação**: Requer validação clínica antes de uso em ambiente real  
**Privacidade**: Dados processados localmente, sem compartilhamento externo

## 📞 Contato

**Autor**: Perseu Padre de Macedo  
**Email**: perseupadre@gmail.com  
**Projeto**: MVP 2 - PUC-Rio  
**Notebook Colab**: [Link direto](https://colab.research.google.com/drive/1-Ao-A9NNUVXYET737x-paAA6xFEKzOIr?usp=sharing)
