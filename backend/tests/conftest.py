"""
Este arquivo configura os testes automatizados para validar
a performance do modelo de Machine Learning.
"""

import pytest
import sys
import os

# Configurações básicas do pytest
def pytest_configure(config):
    """
    🔧 Configuração inicial do pytest
    """
    config.addinivalue_line(
        "markers", "performance: marca testes de performance do modelo"
    )

@pytest.fixture(scope="session")
def model_test_environment():
    """
    Fixture para configurar o ambiente de teste do modelo
    """
    print("\n🔧 Configurando ambiente de teste...")
    
    # Verifica se os arquivos necessários existem
    base_dir = os.path.dirname(os.path.dirname(__file__))
    model_dir = os.path.join(base_dir, 'trained_model')
    
    required_files = [
        os.path.join(model_dir, 'best_model_pipeline.joblib'),
        os.path.join(model_dir, 'feature_columns.joblib')
    ]
    
    missing_files = []
    for file_path in required_files:
        if not os.path.exists(file_path):
            missing_files.append(file_path)
    
    if missing_files:
        pytest.fail(f"❌ Arquivos do modelo não encontrados: {missing_files}")
    
    print("✅ Ambiente de teste configurado com sucesso!")
    
    return {
        "model_dir": model_dir,
        "base_dir": base_dir,
        "status": "ready"
    }
