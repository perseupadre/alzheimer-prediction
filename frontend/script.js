/**
 * Sistema de Predição de Risco de Alzheimer
 * Interface web para análise de fatores de risco baseada em machine learning
 */

// Configurações globais
const CONFIG = {
    PESO_BASE: 1,
    PESO_COGNITIVO: 3,
    PESO_FUNCIONAL: 2,
    THRESHOLD_RISCO: 40,
    MAX_HISTORICO: 50,
    PONTUACAO_MAXIMA: 27
};

// Mapeamentos e traduções
const TRADUCOES = {
    'CardiovascularDisease': 'Doença cardiovascular',
    'Diabetes': 'Diabetes',
    'Depression': 'Depressão',
    'HeadInjury': 'Traumatismo craniano',
    'Hypertension': 'Hipertensão',
    'MemoryComplaints': 'Queixas de memória',
    'BehavioralProblems': 'Problemas comportamentais',
    'Confusion': 'Confusão mental',
    'Disorientation': 'Desorientação',
    'PersonalityChanges': 'Mudanças de personalidade',
    'DifficultyCompletingTasks': 'Dificuldade para completar tarefas',
    'Forgetfulness': 'Esquecimento'
};

/**
 * Classe principal para análise de risco de Alzheimer
 */
class AlzheimerPredictor {
    constructor() {
        this.elementos = this.initializeElements();
        this.setupEventListeners();
        this.carregarHistorico();
    }

    initializeElements() {
        return {
            form: document.getElementById('predictionForm'),
            result: document.getElementById('predictionResult'),
            error: document.getElementById('errorMessage'),
            historyBtn: document.getElementById('loadHistoryBtn'),
            historyList: document.getElementById('historyList'),
            height: document.getElementById('Height'),
            weight: document.getElementById('Weight'),
            bmi: document.getElementById('BMI')
        };
    }

    setupEventListeners() {
        this.elementos.form.addEventListener('submit', (e) => this.handleSubmit(e));
        this.elementos.historyBtn.addEventListener('click', () => this.carregarHistorico());
        this.elementos.height.addEventListener('input', () => this.calcularIMC());
        this.elementos.weight.addEventListener('input', () => this.calcularIMC());
    }

    calcularIMC() {
        const altura = parseFloat(this.elementos.height.value);
        const peso = parseFloat(this.elementos.weight.value);
        
        if (altura && peso && altura > 0) {
            const imc = peso / (altura * altura);
            this.elementos.bmi.value = imc.toFixed(2);
        }
    }

    analisarRisco(dados) {
        let pontos = 0;
        let fatoresRisco = [];

        // Análise por categorias
        pontos += this.analisarDemografia(dados, fatoresRisco);
        pontos += this.analisarAntropometria(dados, fatoresRisco);
        pontos += this.analisarEstiloVida(dados, fatoresRisco);
        pontos += this.analisarCondicoesMedicas(dados, fatoresRisco);
        pontos += this.analisarExames(dados, fatoresRisco);
        pontos += this.analisarCognicao(dados, fatoresRisco);
        pontos += this.analisarSintomas(dados, fatoresRisco);

        return { pontos, fatoresRisco };
    }

    analisarDemografia(dados, fatoresRisco) {
        const idade = parseInt(dados.Age) || 0;
        if (idade >= 70) {
            fatoresRisco.push(`Idade avançada: ${idade} anos`);
            return CONFIG.PESO_BASE;
        }
        return 0;
    }

    analisarAntropometria(dados, fatoresRisco) {
        const imc = parseFloat(dados.BMI) || 0;
        if (imc < 18.5 || imc > 30) {
            fatoresRisco.push(`IMC inadequado: ${imc.toFixed(1)}`);
            return CONFIG.PESO_BASE;
        }
        return 0;
    }

    analisarEstiloVida(dados, fatoresRisco) {
        let pontos = 0;
        
        if (dados.Smoking === '1') {
            pontos += CONFIG.PESO_BASE;
            fatoresRisco.push('Tabagismo');
        }
        
        if (dados.FamilyHistoryAlzheimers === '1') {
            pontos += CONFIG.PESO_BASE;
            fatoresRisco.push('Histórico familiar de Alzheimer');
        }
        
        const atividade = parseFloat(dados.PhysicalActivity) || 0;
        if (atividade < 2) {
            pontos += CONFIG.PESO_BASE;
            fatoresRisco.push('Atividade física insuficiente');
        }
        
        const sono = parseFloat(dados.SleepQuality) || 10;
        if (sono < 6) {
            pontos += CONFIG.PESO_BASE;
            fatoresRisco.push('Qualidade do sono ruim');
        }
        
        return pontos;
    }

    analisarCondicoesMedicas(dados, fatoresRisco) {
        let pontos = 0;
        const condicoes = ['CardiovascularDisease', 'Diabetes', 'Depression', 'HeadInjury', 'Hypertension'];
        
        condicoes.forEach(condicao => {
            if (dados[condicao] === '1') {
                pontos += CONFIG.PESO_BASE;
                fatoresRisco.push(TRADUCOES[condicao] || condicao);
            }
        });
        
        return pontos;
    }

    analisarExames(dados, fatoresRisco) {
        let pontos = 0;
        
        const pressaoSist = parseInt(dados.SystolicBP) || 0;
        const pressaoDiast = parseInt(dados.DiastolicBP) || 0;
        if (pressaoSist > 140 || pressaoDiast > 90) {
            pontos += CONFIG.PESO_BASE;
            fatoresRisco.push('Pressão arterial elevada');
        }

        const colesterol = parseFloat(dados.CholesterolTotal) || 0;
        if (colesterol > 240) {
            pontos += CONFIG.PESO_BASE;
            fatoresRisco.push('Colesterol total elevado');
        }
        
        return pontos;
    }

    analisarCognicao(dados, fatoresRisco) {
        let pontos = 0;
        
        const mmse = parseFloat(dados.MMSE) || 30;
        if (mmse < 24) {
            pontos += CONFIG.PESO_COGNITIVO;
            fatoresRisco.push(`MMSE baixo: ${mmse}/30`);
        }

        const adl = parseFloat(dados.ADL) || 10;
        if (adl < 8) {
            pontos += CONFIG.PESO_FUNCIONAL;
            fatoresRisco.push(`ADL baixo: ${adl}/10`);
        }

        const funcional = parseFloat(dados.FunctionalAssessment) || 10;
        if (funcional < 7) {
            pontos += CONFIG.PESO_FUNCIONAL;
            fatoresRisco.push(`Avaliação funcional baixa: ${funcional}/10`);
        }
        
        return pontos;
    }

    analisarSintomas(dados, fatoresRisco) {
        let pontos = 0;
        const sintomas = ['MemoryComplaints', 'BehavioralProblems', 'Confusion', 'Disorientation', 
                         'PersonalityChanges', 'DifficultyCompletingTasks', 'Forgetfulness'];
        
        sintomas.forEach(sintoma => {
            if (dados[sintoma] === '1') {
                pontos += CONFIG.PESO_BASE;
                fatoresRisco.push(TRADUCOES[sintoma] || sintoma);
            }
        });
        
        return pontos;
    }

    determinarDiagnostico(pontos) {
        const porcentagem = Math.round((pontos / CONFIG.PONTUACAO_MAXIMA) * 100);
        
        if (porcentagem >= CONFIG.THRESHOLD_RISCO) {
            return {
                resultado: 'Predisposição para Alzheimer: Sim - Risco Considerável',
                porcentagem,
                cor: '#dc3545',
                icon: '⚠️'
            };
        } else {
            return {
                resultado: 'Predisposição para Alzheimer: Não - Risco Baixo',
                porcentagem,
                cor: '#28a745',
                icon: '✅'
            };
        }
    }

    gerarResumo(dados) {
        const resumo = [];
        
        if (dados.Age) resumo.push(`Idade: ${dados.Age} anos`);
        if (dados.Gender) resumo.push(`Gênero: ${dados.Gender === '1' ? 'Feminino' : 'Masculino'}`);
        if (dados.MMSE) resumo.push(`MMSE: ${dados.MMSE}/30`);
        if (dados.ADL) resumo.push(`ADL: ${dados.ADL}/10`);
        if (dados.FunctionalAssessment) resumo.push(`Func: ${dados.FunctionalAssessment}/10`);
        
        // Adicionar sintomas principais se presentes
        if (dados.MemoryComplaints === '1') resumo.push('Queixas de memória');
        if (dados.BehavioralProblems === '1') resumo.push('Problemas comportamentais');
        if (dados.FamilyHistoryAlzheimers === '1') resumo.push('Histórico familiar');
        
        return resumo.slice(0, 5).join(', ');
    }

    gerarRecomendacoes(diagnostico) {
        if (diagnostico.porcentagem >= CONFIG.THRESHOLD_RISCO) {
            return [
                'Consulte um neurologista para avaliação detalhada',
                'Realize exames neuropsicológicos específicos',
                'Mantenha atividade física regular',
                'Estimule atividades cognitivas (leitura, jogos, aprendizado)'
            ];
        } else {
            return [
                'Continue com hábitos saudáveis',
                'Mantenha atividade física regular',
                'Durma bem (7-9 horas por noite)',
                'Tenha uma dieta equilibrada'
            ];
        }
    }

    salvarNoHistorico(dados, diagnostico, fatoresRisco) {
        const historico = JSON.parse(localStorage.getItem('alzheimerHistorico') || '[]');
        const agora = new Date();
        
        const novoRegistro = {
            id: 'P' + String(historico.length + 1).padStart(4, '0'),
            data: agora.toLocaleString('pt-BR'),
            diagnostico: diagnostico.resultado,
            porcentagem: diagnostico.porcentagem,
            resumo: this.gerarResumo(dados),
            recomendacoes: this.gerarRecomendacoes(diagnostico),
            medico: dados.DoctorInCharge || 'Não informado'
        };
        
        historico.unshift(novoRegistro);
        
        // Manter apenas os últimos 50 registros
        if (historico.length > CONFIG.MAX_HISTORICO) {
            historico.splice(CONFIG.MAX_HISTORICO);
        }
        
        localStorage.setItem('alzheimerHistorico', JSON.stringify(historico));
        return novoRegistro.id;
    }

    exibirResultado(analise, diagnostico, pacienteId) {
        this.elementos.result.innerHTML = `
            <div style="border: 1px solid ${diagnostico.cor}; border-radius: 8px; padding: 15px; margin: 15px 0; background: white;">
                <div style="margin-bottom: 10px;">
                    <strong style="color: ${diagnostico.cor}; font-size: 1.1em;">
                        ${diagnostico.icon} ${diagnostico.resultado}
                    </strong>
                </div>
                
                <div style="margin: 10px 0;">
                    <strong>👤 ID do Paciente:</strong> ${pacienteId} | 
                    <strong>📊 Porcentagem de Risco:</strong> ${diagnostico.porcentagem}%
                </div>

                ${analise.fatoresRisco.length > 0 ? `
                    <div style="background: #fff3cd; padding: 10px; border-radius: 5px; margin: 10px 0;">
                        <strong>⚠️ Fatores de Risco (${analise.fatoresRisco.length}):</strong>
                        <p style="margin: 5px 0; font-size: 0.9em;">
                            ${analise.fatoresRisco.slice(0, 6).join('. ')}.${analise.fatoresRisco.length > 6 ? ` E mais ${analise.fatoresRisco.length - 6} fatores.` : ''}
                        </p>
                    </div>
                ` : `
                    <div style="background: #d1edff; padding: 10px; border-radius: 5px; margin: 10px 0;">
                        <strong>✅ Perfil de Baixo Risco:</strong> Nenhum fator significativo identificado.
                    </div>
                `}

                <div style="background: #f8f9fa; padding: 10px; border-radius: 5px; margin-top: 10px; font-size: 0.85em; color: #666;">
                    <strong>Sistema:</strong> ${CONFIG.PONTUACAO_MAXIMA} características analisadas | Threshold: ≥${CONFIG.THRESHOLD_RISCO}% = Considerável
                    <br><em>Salvo no histórico com ID: ${pacienteId}</em>
                </div>
            </div>
        `;
        this.elementos.result.style.display = 'block';
    }

    carregarHistorico() {
        const historico = JSON.parse(localStorage.getItem('alzheimerHistorico') || '[]');
        
        if (historico.length === 0) {
            this.elementos.historyList.innerHTML = `
                <div style="text-align: center; padding: 20px; color: #666;">
                    <p>📝 Nenhum registro encontrado no histórico.</p>
                    <p><small>Os resultados das predições aparecerão aqui automaticamente.</small></p>
                </div>
            `;
            return;
        }

        this.elementos.historyList.innerHTML = `
            <div style="margin: 20px 0;">
                <h3>📋 Últimos ${historico.length} Registros</h3>
                ${historico.map(registro => `
                    <div style="border: 1px solid #ddd; border-radius: 8px; padding: 15px; margin: 10px 0; background: white;">
                        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px;">
                            <h4 style="margin: 0; color: #333;">👤 ${registro.id}</h4>
                            <span style="font-size: 0.9em; color: #666;">📅 ${registro.data}</span>
                        </div>
                        
                        <div style="margin: 10px 0;">
                            <strong>🏥 Diagnóstico:</strong> ${registro.diagnostico} (${registro.porcentagem}%)
                        </div>
                        
                        <div style="margin: 10px 0;">
                            <strong>📊 Resumo:</strong> ${registro.resumo}
                        </div>
                        
                        <div style="margin: 10px 0;">
                            <strong>👨‍⚕️ Médico:</strong> ${registro.medico}
                        </div>
                        
                        <div style="background: #f8f9fa; padding: 10px; border-radius: 5px; margin-top: 10px;">
                            <strong>💡 Recomendações:</strong>
                            <ul style="margin: 5px 0; padding-left: 20px;">
                                ${registro.recomendacoes.map(rec => `<li>${rec}</li>`).join('')}
                            </ul>
                        </div>
                    </div>
                `).join('')}
            </div>
        `;
    }

    handleSubmit(e) {
        e.preventDefault();
        
        // Limpar mensagens anteriores
        this.elementos.result.style.display = 'none';
        this.elementos.error.style.display = 'none';

        try {
            // Coletar e validar dados
            const dados = this.coletarDados();
            this.validarDados(dados);

            // Analisar risco
            const analise = this.analisarRisco(dados);
            const diagnostico = this.determinarDiagnostico(analise.pontos);

            // Salvar e exibir resultado
            const pacienteId = this.salvarNoHistorico(dados, diagnostico, analise.fatoresRisco);
            this.exibirResultado(analise, diagnostico, pacienteId);

            // Scroll suave para o resultado
            this.elementos.result.scrollIntoView({ behavior: 'smooth', block: 'start' });

            // Log para debug
            console.log('Análise concluída:', { dados, analise, diagnostico, pacienteId });

        } catch (err) {
            this.exibirErro(err.message);
            console.error('Erro na análise:', err);
        }
    }

    coletarDados() {
        const formData = new FormData(this.elementos.form);
        return Object.fromEntries(formData);
    }

    validarDados(dados) {
        const camposObrigatorios = ['Age', 'Gender', 'Ethnicity', 'EducationLevel', 'Height', 'Weight'];
        const camposFaltando = camposObrigatorios.filter(campo => 
            !dados[campo] || dados[campo].toString().trim() === ''
        );

        if (camposFaltando.length > 0) {
            throw new Error(`Por favor, preencha os campos obrigatórios: ${camposFaltando.join(', ')}`);
        }
    }

    exibirErro(mensagem) {
        this.elementos.error.innerHTML = `
            <div style="background: #f8d7da; color: #721c24; padding: 15px; border-radius: 8px; margin: 10px 0; border: 1px solid #f5c6cb;">
                <strong>❌ Erro:</strong> ${mensagem}
            </div>
        `;
        this.elementos.error.style.display = 'block';
    }
}

// Inicialização quando DOM estiver carregado
document.addEventListener('DOMContentLoaded', () => {
    new AlzheimerPredictor();
});