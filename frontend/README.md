# 🧠 Frontend - Interface de Predição de Alzheimer

Interface web responsiva para sistema de predição de risco de Alzheimer. Integra 32 features médicas com análise simplificada e histórico automático.

## � Características Principais

**� Sistema Completo**: 32 features médicas categorizadas  
**🎯 Diagnóstico Simplificado**: "Não - Risco Baixo" ou "Sim - Risco Considerável"  
**📈 Análise Inteligente**: Pesos especiais para fatores cognitivos  
**🏥 Histórico Automático**: Salva predições com ID e recomendações  
**📱 Design Responsivo**: Funciona em desktop, tablet e mobile

## 📋 Features Organizadas (32 variáveis)

**Obrigatórias (7)**: Dados demográficos + antropométricos  
**Opcionais (25)**: Estilo vida, condições médicas, exames, avaliação cognitiva, sintomas neurológicos

### Pesos Especiais no Cálculo
- **MMSE**: peso x3 (mais crítico)
- **ADL**: peso x2 
- **Avaliação Funcional**: peso x2
- **Demais**: peso x1

## 🎯 Como Usar

### 1. Preenchimento Mínimo
Apenas 7 campos obrigatórios: Idade, Gênero, Etnia, Educação, Altura, Peso (BMI calculado automaticamente)

### 2. Dados Opcionais
25 campos adicionais melhoram precisão da análise

### 3. Resultado Instantâneo
- **Não - Risco Baixo** (<40%): ✅ Verde
- **Sim - Risco Considerável** (≥40%): ⚠️ Vermelho
- Lista fatores de risco identificados
- Recomendações clínicas personalizadas

### 4. Histórico Automático
- ID único do paciente (P0001, P0002...)
- Data/hora da consulta
- Resumo dos dados principais
- Médico responsável (opcional)

## ⚙️ Arquitetura JavaScript (Refatorada v2.1)

### Classe `AlzheimerPredictor` (ES6+)
```javascript
class AlzheimerPredictor {
    constructor() {
        this.elementos = this.initializeElements();
        this.setupEventListeners();
        this.carregarHistorico();
    }
    
    // Métodos especializados por categoria
    analisarDemografia(dados, fatoresRisco) { ... }
    analisarCognicao(dados, fatoresRisco) { ... }
    analisarSintomas(dados, fatoresRisco) { ... }
    
    // Controle de fluxo
    handleSubmit(e) { ... }
    exibirResultado(analise, diagnostico, pacienteId) { ... }
}
```

### Configurações Centralizadas
```javascript
const CONFIG = {
    PESO_BASE: 1,
    PESO_COGNITIVO: 3,    // Peso especial para MMSE
    PESO_FUNCIONAL: 2,    // Peso especial para ADL/Funcional
    THRESHOLD_RISCO: 40,
    MAX_HISTORICO: 50,
    PONTUACAO_MAXIMA: 27
};
```

### Boas Práticas Implementadas
- ✅ **Orientação a Objetos**: Encapsulamento em classe ES6
- ✅ **Single Responsibility**: Métodos focados e pequenos  
- ✅ **Configuração Centralizada**: Constants object
- ✅ **Error Handling**: Tratamento consistente de erros
- ✅ **Event Management**: Listeners organizados no constructor
### Manutenibilidade Aprimorada
```javascript
// Análise por categorias (método organizado)
analisarRisco(dados) {
    let pontos = 0;
    let fatoresRisco = [];

    // Análise modularizada por categoria
    pontos += this.analisarDemografia(dados, fatoresRisco);
    pontos += this.analisarAntropometria(dados, fatoresRisco);
    pontos += this.analisarEstiloVida(dados, fatoresRisco);
    pontos += this.analisarCondicoesMedicas(dados, fatoresRisco);
    pontos += this.analisarExames(dados, fatoresRisco);
    pontos += this.analisarCognicao(dados, fatoresRisco);
    pontos += this.analisarSintomas(dados, fatoresRisco);

    return { pontos, fatoresRisco };
}
```
```

### Histórico com Encapsulamento
```javascript
// Método da classe (this.elementos referenciado automaticamente)
salvarNoHistorico(dados, diagnostico, fatoresRisco) {
    const historico = JSON.parse(localStorage.getItem('alzheimerHistorico') || '[]');
    const novoRegistro = {
        id: 'P' + String(historico.length + 1).padStart(4, '0'),
        data: new Date().toLocaleString('pt-BR'),
        diagnostico: diagnostico.resultado,
        porcentagem: diagnostico.porcentagem,
        resumo: this.gerarResumo(dados),
        recomendacoes: this.gerarRecomendacoes(diagnostico),
        medico: dados.DoctorInCharge || 'Não informado'
    };
    
    historico.unshift(novoRegistro);
    if (historico.length > CONFIG.MAX_HISTORICO) {
        historico.splice(CONFIG.MAX_HISTORICO);
    }
    localStorage.setItem('alzheimerHistorico', JSON.stringify(historico));
    return novoRegistro.id;
}
```

### Validação Robusta
```javascript
// Validação com tratamento de erros organizado
validarDados(dados) {
    const camposObrigatorios = ['Age', 'Gender', 'Ethnicity', 'EducationLevel', 'Height', 'Weight'];
    const camposFaltando = camposObrigatorios.filter(campo => 
        !dados[campo] || dados[campo].toString().trim() === ''
    );

    if (camposFaltando.length > 0) {
        throw new Error(`Por favor, preencha os campos obrigatórios: ${camposFaltando.join(', ')}`);
    }
}
```

## 📊 Recomendações Automáticas

**Predisposição para Alzheimer: Sim - Risco Considerável (≥40%)**:  
1. Consulte neurologista para avaliação detalhada
2. Realize exames neuropsicológicos específicos  
3. Mantenha atividade física regular
4. Estimule atividades cognitivas

**Predisposição para Alzheimer: Não - Risco Baixo (<40%)**:  
1. Continue hábitos saudáveis
2. Mantenha atividade física  
3. Durma bem (7-9h por noite)
4. Dieta equilibrada


- **Dados Sensíveis**: Armazenados apenas localmente no navegador

### Limitações
- **Não é Diagnóstico**: Apenas indicador de risco baseado em padrões estatísticos
- **Modelo Pré-treinado**: Parâmetros fixos (não re-treina automaticamente)  
- **Base de Conhecimento**: Baseado no dataset específico do Kaggle
- **Predições Ilimitadas**: Processa QUALQUER novo paciente em tempo real


## 🔄 Atualizações e Manutenção

### Versão Atual: 1.0
- ✅ **Processamento Ilimitado**: Aceita novos pacientes indefinidamente
- ✅ **Predições em Tempo Real**: Resultado instantâneo para cada caso
- ✅ **32 Features Médicas**: Análise completa e categorizada
- ✅ **Histórico Persistente**: Salva todas as consultas automaticamente
- ✅ **Interface Responsiva**: Otimizada para qualquer dispositivo

### Próximas Versões
- Integração com API backend (opcional)
- Exportação de relatórios em PDF
- Gráficos de evolução temporal
- Sistema de alertas por email

## 📚 Documentação Adicional

- **script.js**: Arquitetura orientada a objetos (classe ES6) com boas práticas
- **Kaggle Dataset**: [Alzheimer's Disease Dataset](https://www.kaggle.com/datasets/rabieelkharoua/alzheimers-disease-dataset)

---

*Sistema desenvolvido para MVP 2 - Engenharia de Software*  
*Versão 1.0 - Julho de 2025*  

1. O frontend é servido automaticamente pelo Flask em `http://localhost:5000`
2. Não requer servidor web separado
3. Arquivos estáticos servidos pela rota `/` do Flask

### Deploy em Produção
1. **Servidor Web**: Configure nginx/Apache para servir arquivos estáticos
2. **HTTPS**: Sempre use HTTPS em produção para dados médicos
3. **CORS**: Configure adequadamente para o domínio de produção
4. **API URL**: Atualize `API_BASE_URL` para o endpoint de produção

### Configuração da API
```javascript
// Desenvolvimento
const API_BASE_URL = 'http://127.0.0.1:5000';

// Produção
const API_BASE_URL = 'https://api.seudominio.com';
```

## 📱 Compatibilidade

### Navegadores Suportados
- **Chrome**: 90+
- **Firefox**: 88+
- **Safari**: 14+
- **Edge**: 90+

### Dispositivos
- **Desktop**: Resolução mínima 1024x768
- **Tablet**: Otimizado para iPad e Android tablets
- **Mobile**: Responsivo para smartphones (375px+)

## ⚠️ Considerações de Segurança

### Dados Sensíveis
- **Não armazena** dados pessoais no frontend
- **Transmissão segura** via HTTPS em produção
- **Validação** tanto no frontend quanto no backend

### Privacidade
- Dados processados localmente no servidor
- Histórico armazenado apenas no banco local
- Sem compartilhamento com terceiros

## 🔗 Integração com Backend

### Endpoints Utilizados
- `POST /predict`: Envio de dados para predição
- `GET /history`: Recuperação do histórico
- `GET /`: Carregamento da página principal

### Formato de Dados
```json
{
    "Age": 65,
    "Gender": 1,
    "Ethnicity": 0,
    "EducationLevel": 2,
    "Height": 1.70,
    "Weight": 75.0,
    "BMI": 25.96,
    "Smoking": 0,
    // ... demais campos
}
```

## 📈 Monitoramento

### Métricas de UX
- Tempo de carregamento da página
- Taxa de preenchimento completo do formulário
- Frequência de uso das funcionalidades
- Erros de validação mais comuns

### Analytics Recomendados
- Google Analytics para uso geral
- Hotjar para heatmaps e sessões
- Monitoramento de erros JavaScript

## 🛠️ Manutenção

### Atualizações Regulares
- Verificar compatibilidade com novos navegadores
- Atualizar dependências JavaScript se aplicável
- Revisar UX baseado em feedback dos usuários
- Manter sincronização com mudanças na API

### Backup
- Código versionado no Git
- Testes regulares em diferentes dispositivos
- Documentação atualizada de mudanças
