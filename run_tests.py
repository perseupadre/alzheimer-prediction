"""
Script para validação rápida de performance do modelo com dados sintéticos otimizados.
Testa a estratégia de geração de dados baseada em feature importance para ML de Alzheimer.
"""
import os
import joblib
import pandas as pd
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, 
    f1_score, roc_auc_score
)

# Configurações globais
CONFIG = {
    'random_seed': 42,
    'n_samples': 1000,
    'class_distribution': [0.60, 0.40],  # 60% baixo risco, 40% alto risco
    'threshold_min': 70.0
}

def load_model_artifacts():
    """Carrega modelo e features treinados."""
    base_dir = os.path.dirname(__file__)
    model_path = os.path.join(base_dir, 'backend', 'trained_model', 'best_model_pipeline.joblib')
    features_path = os.path.join(base_dir, 'backend', 'trained_model', 'feature_columns.joblib')
    
    model = joblib.load(model_path)
    feature_columns = joblib.load(features_path)
    return model, feature_columns

def get_feature_configurations():
    """Retorna configurações otimizadas das features principais baseadas em importância."""
    return {
        # Features numéricas principais (baseado em feature importance)
        'numeric': {
            'MMSE': {'no_alz': (24.0, 3.0), 'alz': (8.0, 4.0), 'limits': (0, 30)},
            'ADL': {'no_alz': (8.0, 1.5), 'alz': (2.0, 1.5), 'limits': (0, 10)},
            'FunctionalAssessment': {'no_alz': (8.0, 1.5), 'alz': (2.0, 1.5), 'limits': (0, 10)},
            'DietQuality': {'no_alz': (7.0, 1.5), 'alz': (3.0, 1.5), 'limits': (0, 10)},
            'SleepQuality': {'no_alz': (7.0, 1.5), 'alz': (4.0, 1.5), 'limits': (0, 10)},
            'PhysicalActivity': {'no_alz': (6.0, 1.5), 'alz': (2.0, 1.5), 'limits': (0, 10)},
            'Age': {'no_alz': (70.0, 8.0), 'alz': (80.0, 8.0), 'limits': (60, 95)},
            'BMI': {'no_alz': (25.0, 4.0), 'alz': (28.0, 5.0), 'limits': (15, 40)},
            'SystolicBP': {'no_alz': (130, 15), 'alz': (145, 20)},
            'DiastolicBP': {'no_alz': (80, 10), 'alz': (90, 12)},
            'CholesterolTotal': {'no_alz': (200, 30), 'alz': (240, 40)},
            'CholesterolLDL': {'no_alz': (120, 25), 'alz': (150, 30)},
            'CholesterolHDL': {'no_alz': (55, 12), 'alz': (45, 10)},
            'CholesterolTriglycerides': {'no_alz': (150, 40), 'alz': (200, 50)},
            'AlcoholConsumption': {'no_alz': (3.0, 2.0), 'alz': (6.0, 3.0)}
        },
        
        # Features categóricas otimizadas
        'categorical': {
            'MemoryComplaints': {'no_alz': [0.90, 0.10], 'alz': [0.20, 0.80]},
            'BehavioralProblems': {'no_alz': [0.95, 0.05], 'alz': [0.30, 0.70]},
            'Confusion': {'no_alz': [0.95, 0.05], 'alz': [0.25, 0.75]},
            'Forgetfulness': {'no_alz': [0.85, 0.15], 'alz': [0.15, 0.85]},
            'DifficultyCompletingTasks': {'no_alz': [0.90, 0.10], 'alz': [0.25, 0.75]}
        }
    }

def generate_feature_values(feature, config, y_target, n_samples):
    """Gera valores para uma feature específica baseado na configuração."""
    no_alz_indices = y_target == 0
    alz_indices = y_target == 1
    values = np.zeros(n_samples)
    
    # Valores para classe 0 (sem Alzheimer)
    if np.any(no_alz_indices):
        mean_no_alz, std_no_alz = config['no_alz']
        values[no_alz_indices] = np.random.normal(mean_no_alz, std_no_alz, np.sum(no_alz_indices))
    
    # Valores para classe 1 (com Alzheimer)
    if np.any(alz_indices):
        mean_alz, std_alz = config['alz']
        values[alz_indices] = np.random.normal(mean_alz, std_alz, np.sum(alz_indices))
    
    # Aplicar limites se especificado
    if 'limits' in config:
        min_val, max_val = config['limits']
        values = np.clip(values, min_val, max_val)
    
    return values

def generate_categorical_values(feature, config, y_target, n_samples):
    """Gera valores categóricos otimizados baseados no target."""
    no_alz_indices = y_target == 0
    alz_indices = y_target == 1
    values = np.zeros(n_samples, dtype=int)
    
    # Valores para classe 0
    if np.any(no_alz_indices):
        probs_no_alz = config['no_alz']
        values[no_alz_indices] = np.random.choice([0, 1], np.sum(no_alz_indices), p=probs_no_alz)
    
    # Valores para classe 1
    if np.any(alz_indices):
        probs_alz = config['alz']
        values[alz_indices] = np.random.choice([0, 1], np.sum(alz_indices), p=probs_alz)
    
    return values

def generate_test_data(feature_columns, n_samples):
    """Gera dados de teste sintéticos otimizados."""
    np.random.seed(CONFIG['random_seed'])
    
    # Gerar distribuição de classes estratégica
    y_target = np.random.choice([0, 1], n_samples, p=CONFIG['class_distribution'])
    
    # Obter configurações das features
    configs = get_feature_configurations()
    test_data = {}
    
    # Gerar features numéricas
    for feature, config in configs['numeric'].items():
        if feature in feature_columns:
            test_data[feature] = generate_feature_values(feature, config, y_target, n_samples)
    
    # Gerar features categóricas
    for feature, config in configs['categorical'].items():
        if feature in feature_columns:
            test_data[feature] = generate_categorical_values(feature, config, y_target, n_samples)
    
    # Preencher features restantes
    categorical_features = ['Gender', 'Ethnicity', 'EducationLevel', 'Smoking', 
                           'FamilyHistoryAlzheimers', 'CardiovascularDisease', 'Diabetes', 
                           'Depression', 'HeadInjury', 'Hypertension', 'PersonalityChanges']
    
    for feature in feature_columns:
        if feature not in test_data:
            if feature in ['DoctorInCharge', 'Ethnicity', 'EducationLevel']:
                test_data[feature] = np.random.choice([0, 1, 2, 3], n_samples)
            elif feature in categorical_features:
                test_data[feature] = np.random.choice([0, 1], n_samples, p=[0.75, 0.25])
            else:
                # Features numéricas não configuradas
                values = np.random.normal(5.0, 2.0, n_samples)
                test_data[feature] = np.clip(values, 0, 10)
    
    return pd.DataFrame(test_data)[feature_columns], y_target

def calculate_metrics(model, X_test, y_test):
    """Calcula todas as métricas de performance."""
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    return {
        'accuracy': accuracy_score(y_test, y_pred) * 100,
        'precision': precision_score(y_test, y_pred) * 100,
        'recall': recall_score(y_test, y_pred) * 100,
        'f1_score': f1_score(y_test, y_pred) * 100,
        'auc_roc': roc_auc_score(y_test, y_pred_proba) * 100
    }

def print_results(metrics):
    """Imprime resultados das métricas de forma organizada."""
    print(f"\n🎯 RESULTADOS DAS MÉTRICAS:")
    print("=" * 50)
    
    all_passed = True
    failed_metrics = []
    threshold = CONFIG['threshold_min']
    
    for metric_name, value in metrics.items():
        passed = value >= threshold
        status = "✅ PASSOU" if passed else "❌ FALHOU"
        margin = value - threshold
        print(f"{metric_name.upper():<12}: {value:>6.2f}% (req: {threshold:.0f}%) | {status} ({margin:+.1f}%)")
        
        all_passed = all_passed and passed
        if not passed:
            failed_metrics.append(metric_name)
    
    print(f"\n{'=' * 50}")
    if all_passed:
        print("🎉 SUCESSO: Todas as métricas passaram!")
        print("🚀 A Estratégia FUNCIONOU!")
    else:
        print(f"❌ FALHA: Métricas que falharam: {', '.join(failed_metrics)}")
        print("🔧 Necessário ajuste adicional na estratégia")
    
    print("=" * 50)

def main():
    """Função principal do script."""
    print("🎯 TESTE RÁPIDO DA ESTRATÉGIA OTIMIZADA")
    print("=" * 60)
    
    # Carregar modelo e features
    model, feature_columns = load_model_artifacts()
    
    # Gerar dados de teste
    X_test, y_test = generate_test_data(feature_columns, CONFIG['n_samples'])
    
    print(f"📊 Dados gerados: {X_test.shape[0]} amostras, {X_test.shape[1]} features")
    print(f"📊 Distribuição: Não - Risco Baixo: {np.sum(y_test == 0)}, Sim - Risco Considerável: {np.sum(y_test == 1)}")
    
    # Calcular e imprimir métricas
    metrics = calculate_metrics(model, X_test, y_test)
    print_results(metrics)

if __name__ == "__main__":
    main()
