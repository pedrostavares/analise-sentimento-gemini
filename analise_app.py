# -*- coding: utf-8 -*- # Adicionado para garantir codificação UTF-8

import streamlit as st
import pandas as pd
import google.generativeai as genai
import io # Para lidar com o arquivo em memória
import time # Para possíveis pausas
import plotly.express as px # Para gráficos
import numpy as np # Para cálculos numéricos (usado no NPS)

# --- Configuração da Página ---
st.set_page_config(
    layout="wide",
    page_title="Análise de Sentimento e Temática - IH",
    page_icon="📊"
)

# --- Inicialização do Estado da Sessão ---
if 'api_key_configured' not in st.session_state: st.session_state.api_key_configured = False
if 'api_key_input_value' not in st.session_state: st.session_state.api_key_input_value = ""
if 'analysis_done' not in st.session_state: st.session_state.analysis_done = False
if 'df_results' not in st.session_state: st.session_state.df_results = None
if 'insights_generated' not in st.session_state: st.session_state.insights_generated = None # Para guardar os insights

# --- Configuração da API Key ---
api_key_source = None
try:
    # Tenta carregar a chave dos segredos do Streamlit (ideal para deploy)
    if hasattr(st, 'secrets') and "GOOGLE_API_KEY" in st.secrets:
        st.session_state.api_key_input_value = st.secrets["GOOGLE_API_KEY"]
        api_key_source = "secrets"
except Exception as e:
    st.sidebar.warning(f"Não foi possível ler os secrets: {e}", icon="⚠️")

# --- Interface da Barra Lateral para API Key ---
st.sidebar.header("Configuração")
# Só mostra o input se a chave NÃO veio dos secrets
if api_key_source != "secrets":
    user_provided_key = st.sidebar.text_input(
        "Insira sua Google API Key aqui:", type="password",
        key="api_key_widget", value=st.session_state.api_key_input_value,
        help="Sua chave da API do Google AI Studio (Gemini)."
    )
    # Atualiza o estado se a chave mudar no input
    if user_provided_key != st.session_state.api_key_input_value:
         st.session_state.api_key_input_value = user_provided_key
         st.session_state.api_key_configured = False # Reseta a configuração ao mudar a chave
         st.session_state.analysis_done = False # Reseta análise se chave mudar
         st.session_state.insights_generated = None # Reseta insights
         st.rerun() # Re-executa para tentar configurar com a nova chave
else:
    st.sidebar.success("API Key carregada dos segredos!", icon="✅")
    # Marca como não configurado inicialmente para forçar a configuração abaixo
    if not st.session_state.api_key_configured:
        st.session_state.api_key_configured = False

# --- Tentativa de Configurar a API e o Modelo ---
model = None
# Tenta configurar se tem uma chave e ainda não foi configurado
if st.session_state.api_key_input_value and not st.session_state.api_key_configured:
    try:
        genai.configure(api_key=st.session_state.api_key_input_value)
        model = genai.GenerativeModel('gemini-1.5-flash') # Ou 'gemini-pro' se preferir
        st.session_state.api_key_configured = True
        if api_key_source != "secrets": # Só mostra sucesso se a chave foi inserida manualmente
            st.sidebar.success("API Key configurada com sucesso!", icon="🔑")
        st.sidebar.caption(f"Modelo Gemini: gemini-1.5-flash") # Mostra o modelo
    except Exception as e:
        st.sidebar.error(f"Erro ao configurar API Key/Modelo. Verifique a chave.", icon="🚨")
        st.session_state.api_key_configured = False
        model = None
# Se já estava configurado, tenta recarregar o modelo (útil se houve erro antes)
elif st.session_state.api_key_configured:
     try:
         model = genai.GenerativeModel('gemini-1.5-flash')
         # Não precisa mostrar mensagem de sucesso de novo aqui
     except Exception as e:
         st.sidebar.error(f"Erro ao recarregar o Modelo: {e}", icon="🚨")
         st.session_state.api_key_configured = False # Marca como não configurado se falhar
         model = None


# --- Prompt Principal REFINADO v4 (Baseado na sua atualização) ---
seu_prompt_completo = """
Persona: Você é uma IA Analista de Feedback de Clientes e Social Listening altamente especializada no setor bancário brasileiro, com profundo conhecimento sobre o Banco Itaú e seu ecossistema de marcas (incluindo, mas não se limitando a: Itaú (Masterbrand), Itaú Personnalité (Alta Renda), Uniclass (Média Renda), Itaú Empresas (PJ), íon (Investimentos), Private (Altíssima Renda), Itaú BBA (Agro, Atacado, Investment Banking), Itubers/Coração Laranja (Endomarketing)). Você compreende produtos financeiros específicos (CDB, LCI, crédito, etc.), jargões do mercado (TED, DOC, Pix, spread) e o contexto de campanhas de marketing que utilizam influenciadores, creators, figuras públicas e patrocínios de grandes eventos (como Rock in Rio, The Town, Miami Open). Sua análise combina rigor na aplicação das regras de classificação com empatia e compreensão contextual para interpretar nuances e casos limítrofes de forma próxima à humana.

Objetivo:
1.	Primário: Classificar CADA mensagem recebida em Português do Brasil (Pt-BR) em UMA categoria de Sentimento (Positivo, Negativo, Neutro, Não Classificado) e UMA categoria Temática (conforme lista e regras de prioridade), aderindo estritamente às definições e minimizando 'Não Classificado'.
2.	Secundário: (Este prompt foca APENAS na classificação. A geração de insights será feita em outra etapa).

Contexto Geral: As mensagens são de clientes e público geral interagindo com posts e conteúdos do Itaú e suas submarcas, cobrindo produtos, serviços, atendimento, plataformas digitais, campanhas publicitárias, influenciadores associados, patrocínios e a imagem geral da instituição. Assuma que reações curtas são contextuais ao post original e interprete termos técnicos e siglas do setor financeiro corretamente.

=== REGRAS GERAIS E DE OURO ===
1.	Análise Dupla Obrigatória: Sentimento + Tema para cada mensagem.
2.	Priorize P/N/Neutro: Só use 'Não Classificado' como ÚLTIMO recurso absoluto para mensagens que se encaixam estritamente nas definições de NC.
3.	Vinculação NC: Se Sentimento = Não Classificado, Tema = Não Classificado (Tema).
4.	Regra Crítica - Menção Isolada (Única ou Múltipla): Mensagem APENAS com menção (@ ou []) = Sentimento: Positivo, Tema: Interação Social e Engajamento. SEMPRE RESPONDA ASSIM PARA ESTES CASOS.

=== DEFINIÇÕES DE SENTIMENTO (Escolha UMA) ===
1.	Positivo: Expressa satisfação, apoio, entusiasmo, gratidão genuína, apreciação (mesmo moderada "interessante"), concordância, ou engajamento via compartilhamento/marcação.
    *   Indicadores: Menções isoladas (@/[]); Elogios ("Amei", "Top"), agradecimentos ("Obg"), apoio ("Parabéns"), apreciação ("Belo post", "Interessante"), concordância ("Isso"); Emojis claramente positivos (😍, ❤️, 👍, 🎉); Combinações Texto/Emoji Positivo.
2.	Negativo: Expressa insatisfação, crítica, raiva, frustração, reclamação, tristeza, sarcasmo óbvio, ou qualquer relato/indicação de problema, falha, erro, golpe, fraude ou experiência ruim.
    *   Indicadores: Críticas ("Péssimo"), relato/indicação forte de problemas ("Não funciona CDB", "Cobrança indevida DOC", "Fui vítima de golpe Pix", "@itau Erro TED"), reclamações ("Atendimento Uniclass ruim"), insatisfação ("Taxa alta BBA"), frustração (CAIXA ALTA negativa), advertência ("Não recomendo íon"); Sarcasmo óbvio; Emojis claramente negativos (😠, 😡, 👎, 😢). Qualquer menção a golpe, fraude, erro ou problema funcional é Negativa.
3.	Neutro: Busca/fornece informação, observação factual, pergunta, sugestão objetiva ou implícita ("poderia ser melhor") (sem crítica/sarcasmo), expressão de equilíbrio ("ok"), ou reação ambígua/sem forte valência P/N por padrão (emojis como 🙏, 😂, 🤔, 👀 sem contexto forte).
    *   Indicadores: Emojis ambíguos padrão; Perguntas/Respostas a perguntas ("Como faço LCI?", "?"), expressões de equilíbrio ("Ok"), sugestões implícitas ("Poderia ser melhor o app Personnalité"), pedidos de ajuda claros (sem problema grave), informações factuais ("O Rock in Rio é patrocinado"), sugestões objetivas, respostas curtas factuais ("Entendido"), outros emojis neutros (👌, 🤷‍♀️, 😐); Termos/siglas específicos ("ESG").
4.	Não Classificado: APLICAR SOMENTE E EXCLUSIVAMENTE QUANDO A MENSAGEM ATENDER A UM DESTES CRITÉRIOS:
    1.	Idioma Estrangeiro (predominante).
    2.	Incompreensível (erros graves, digitação aleatória, sem sentido lógico).
    3.	Menção + Texto Incompleto/Ambíguo/Incompreensível (Ex: "@itau p", "@itau 2", "@itau ..." SEM indicar problema forte ou sarcasmo claro).
    4.	Spam/Link Isolado (conteúdo repetitivo óbvio, promoções não relacionadas, URL isolada SEM contexto relevante).
    5.	Totalmente Off-Topic (sem QUALQUER conexão com Itaú, bancos, finanças, eventos/campanhas associadas. Ex: "Receita de bolo").
    6.	Interação Social Pura Textual Isolada (APENAS saudações/despedidas "Bom dia", APENAS risadas textuais "kkkk", "rsrs").

=== DEFINIÇÕES DE TEMA (Escolha UMA - Aplicar Regras de Prioridade Abaixo) ===
***IMPORTANTE: Use EXATAMENTE um dos nomes de Tema 1 a 10 abaixo. Se Sentimento = Não Classificado, Tema = Não Classificado (Tema).***

1.	Marca e Imagem: Percepção geral da marca Itaú ou submarcas (Personnalité, Uniclass, etc.), reputação, campanhas institucionais, patrocínios gerais (Rock in Rio, The Town, etc.). (Sentimento: P/N/Neutro)
2.	Produtos e Serviços (Geral): Sobre cartões, contas, seguros, investimentos (CDB, LCI, íon), crédito (consignado), taxas, políticas, benefícios, contratação, cancelamento. (Sentimento: P/N/Neutro)
3.	Atendimento e Suporte: Sobre canais (agência, telefone, chat), qualidade do suporte, resolução de problemas pelo atendimento (incluindo de submarcas). (Sentimento: P/N/Neutro)
4.	Plataformas Digitais (App/Site/ATM): Feedback sobre usabilidade, design, functionalities (PIX, TED, DOC, login), performance de app (Itaú, Personnalité, íon, Empresas), site, caixas eletrônicos. (Sentimento: P/N/Neutro)
5.	Figuras Públicas e Representantes: Foco em atletas patrocinados, influenciadores, creators, "laranjinhas", executivos, etc., associados a campanhas ou à marca. (Sentimento: P/N/Neutro)
6.	Eventos e Campanhas Específicas: Discussões focadas em evento/campanha nomeado (logística, experiência, tema), diferente do patrocínio geral. (Sentimento: P/N/Neutro)
7.	Segurança e Fraude: Sobre golpes, fraudes, segurança da conta, phishing, roubos relacionados a contas/cartões. (Sentimento: Geralmente Negativo, pode ser Neutro)
8.	Interação Social e Engajamento: Baixa prioridade. Usar para: Menções isoladas (@/[] - Positivo); Emojis isolados P/N/Neutro sem outro tema claro.
9.	Solicitação/Dúvida/Sugestão (Transversal): Prioridade baixa/média. Usar quando o FOCO PRINCIPAL é a pergunta/pedido/sugestão em si, e não se encaixa melhor em temas mais específicos. (Sentimento: Geralmente Neutro)
10.	Não Classificado (Tema): Exclusivamente quando Sentimento = Não Classificado.

=== REGRAS DE PRIORIDADE PARA TEMAS ===
Aplique na seguinte ordem. Se a mensagem se encaixar em múltiplos temas, escolha o primeiro da lista que se aplicar:
1.	Segurança e Fraude: (Prioridade Máxima) Se mencionar golpe, fraude, segurança.
2.	Plataformas Digitais (App/Site/ATM): Se o feedback for especificamente sobre essas plataformas.
3.	Atendimento e Suporte: Se o foco for a interação com canais de atendimento.
4.	Produtos e Serviços (Geral): Se sobre características, taxas, contratação/cancelamento de produtos/serviços.
5.	Eventos e Campanhas Específicas: Se claramente focado em um evento/campanha nomeado.
6.	Figuras Públicas e Representantes: Se o foco principal for a pessoa/representante.
7.	Marca e Imagem: Para comentários gerais sobre a marca/reputação/patrocínios gerais.
8.	Solicitação/Dúvida/Sugestão (Transversal): Se o foco principal for a pergunta/pedido em si.
9.	Interação Social e Engajamento: Para menções isoladas, emojis isolados sem outro tema forte. (Prioridade Mínima antes de NC).
10.	Não Classificado (Tema): Apenas se Sentimento = Não Classificado.

=== INSTRUÇÕES ADICIONAIS DE CLASSIFICAÇÃO ===
*   Formato de Resposta: EXATAMENTE DUAS LINHAS, SEMPRE:
    Sentimento: [Nome Exato da Categoria de Sentimento]
    Tema: [Nome Exato da Categoria de Tema]
    (Não inclua NADA MAIS, nem explicações, nem markdown, nem cabeçalhos).
*   Aplicar Prioridade de Tema: Siga estritamente as regras de prioridade.
*   Detectar Sarcasmo: Tentar identificar e classificar como Negativo. Se detectado, o tema deve seguir a prioridade normal (ex: Sarcasmo sobre app -> Negativo / Plataformas Digitais).
*   Menções + Texto: Seguir o TEXTO se CLARO e COMPLETO (aplicando prioridade de tema). Se FORTE INDICADOR DE PROBLEMA -> Negativo/Tema Relevante. Se INCOMPLETO/AMBÍGUO/INCOMPREENSÍVEL (sem indicar problema/sarcasmo) -> Não Classificado/Não Classificado (Tema).
*   Emojis Ambíguos: Padrão Neutro (🙏, 😂, 🤔, 👀) sem contexto forte.
*   Valência Leve: "Interessante" -> Positivo / Tema relevante (ex: Marca e Imagem). "Ok", "Poderia ser melhor" -> Neutro / Tema relevante (ex: Plataformas Digitais se sobre app).
*   Perguntas/Respostas: Geralmente Neutro / Tema conforme prioridade (ou Solicitação/Dúvida/Sugestão se genérico).
*   Emojis Mistos: Prioridade para determinar Sentimento: Negativo > Positivo > Neutro.
*   Ênfase (!!!, ???): Modifica/reforça o sentimento base do texto.
*   Mensagens Mistas: Classifique pelo elemento PREDOMINANTE ou mais FORTE (Reclamação/Problema/Fraude/Sarcasmo > outros; Pergunta Clara > outros), aplicando prioridade de tema.

Agora, classifique a seguinte mensagem:
{comment}
"""

# --- Prompt para Geração de Insights (NOVO) ---
prompt_geracao_insights = """
Persona: Você é um Analista de Social Listening Sênior, especializado no Banco Itaú e seu ecossistema. Sua tarefa é interpretar um resumo de dados de classificação de sentimentos e temas de comentários de clientes/público e gerar insights acionáveis.

Contexto: Você recebeu um resumo da análise de {total_comentarios_analisados} comentários. As classificações foram feitas seguindo critérios específicos para o Itaú (Sentimento: Positivo, Negativo, Neutro, NC; Temas: Marca, Produtos, Atendimento, Plataformas, Figuras Públicas, Eventos, Segurança, Interação, Solicitação, NC).

Dados de Resumo Fornecidos:
*   Distribuição Geral de Sentimentos (Total: {total_comentarios_analisados}):
    - Positivo: {count_pos} ({perc_pos:.1f}%)
    - Negativo: {count_neg} ({perc_neg:.1f}%)
    - Neutro: {count_neu} ({perc_neu:.1f}%)
    - Não Classificado / Erros: {count_nc_err} ({perc_nc_err:.1f}%)
*   Top 5 Temas Mais Comentados (Excluindo Interação Social, NC e Erros. Total destes temas: {total_temas_insights}):
{top_temas_formatado}
*   Top 3 Temas Associados ao Sentimento NEGATIVO (Total de comentários negativos com tema válido: {total_temas_neg}):
{top_temas_negativos_formatado}

Tarefa: Com base EXCLUSIVAMENTE nos dados de resumo fornecidos acima, elabore um bloco conciso de "Insights e Percepções Acionáveis". Organize sua resposta usando os seguintes tópicos em Markdown:

### Principais Destaques Positivos:
*   (Comente a proporção de comentários positivos. Se houver dados nos Top Temas, relacione o sentimento positivo a algum tema específico, se possível inferir indiretamente. Ex: "Alto volume positivo pode estar ligado a X tema, se este for predominante.")

### Principais Pontos de Atenção (Negativos):
*   (Comente a proporção de comentários negativos. **Crucialmente, foque nos 'Top 3 Temas Associados ao Sentimento NEGATIVO'**. Identifique as principais áreas de reclamação/crítica. Chame atenção para possíveis "telhados de vidro" ou problemas recorrentes nesses temas.)

### Oportunidades e Sugestões:
*   (Analise a proporção de comentários Neutros. Se o tema 'Solicitação/Dúvida/Sugestão' aparecer nos Top Temas, indique oportunidade de esclarecimento ou melhoria. Se temas negativos recorrentes aparecerem, sugira investigação ou ação específica para mitigar.)

### Observações Gerais:
*   (Faça um balanço geral. Comente se a distribuição de sentimentos parece saudável ou preocupante. Mencione se a proporção de 'Não Classificado/Erros' é alta, indicando possíveis problemas na coleta ou classificação. Destaque algum tema específico que dominou a conversa, se for o caso.)

Instruções Adicionais:
*   Seja direto e focado em insights que possam gerar ações para o Itaú.
*   Baseie-se APENAS nos dados fornecidos no resumo. Não invente informações ou temas não listados.
*   Se algum dado crucial estiver faltando ou for insuficiente (ex: muito poucos comentários negativos para tirar conclusões sobre temas negativos), mencione essa limitação.
*   Mantenha a linguagem profissional e analítica.
*   Use bullet points (*) para listar os insights dentro de cada tópico.
"""

# --- Listas de Categorias Válidas (ATUALIZADAS COM BASE NO NOVO PROMPT) ---
categorias_sentimento_validas = ["Positivo", "Negativo", "Neutro", "Não Classificado"]
categorias_tema_validas = [
    "Marca e Imagem",
    "Produtos e Serviços (Geral)",
    "Atendimento e Suporte",
    "Plataformas Digitais (App/Site/ATM)",
    "Figuras Públicas e Representantes",
    "Eventos e Campanhas Específicas",
    "Segurança e Fraude",
    "Interação Social e Engajamento",
    "Solicitação/Dúvida/Sugestão (Transversal)",
    "Não Classificado (Tema)" # Este é o tema para quando o sentimento é Não Classificado
]
categorias_erro = ["Erro Parsing", "Erro API"]
categorias_erro_tema_especifico = ["Erro API (Timeout)", "Erro API (Geral)", "Erro API (Modelo não iniciado)", "Erro API (Conteúdo Bloqueado)"]
todas_categorias_erro = list(set(categorias_erro + categorias_erro_tema_especifico))

# Categorias a serem excluídas dos cálculos de NPS e gráficos principais
categorias_excluir_sentimento = ["Não Classificado"] + todas_categorias_erro
# Exclui NC, Erros e Interação Social dos gráficos/insights principais de TEMAS de negócio
categorias_excluir_tema = ["Não Classificado (Tema)", "Interação Social e Engajamento"] + todas_categorias_erro


# --- Função para Analisar um Comentário ---
def analisar_comentario(comentario, modelo_gemini):
    """Envia um comentário para a API Gemini e retorna o sentimento e tema classificados."""
    # Retorna NC para comentários vazios ou inválidos
    if not comentario or not isinstance(comentario, str) or comentario.strip() == "":
        return "Não Classificado", "Não Classificado (Tema)"
    # Retorna erro se o modelo não foi inicializado
    if not modelo_gemini:
        return "Erro API", "Erro API (Modelo não iniciado)"

    # Preenche o prompt com o comentário atual
    prompt_com_comentario = seu_prompt_completo.format(comment=comentario)
    try:
        # Configurações de segurança (relaxadas para evitar bloqueios comuns em feedback)
        safety_settings = {
            "HARM_CATEGORY_HARASSMENT": "BLOCK_NONE",
            "HARM_CATEGORY_HATE_SPEECH": "BLOCK_NONE",
            "HARM_CATEGORY_SEXUALLY_EXPLICIT": "BLOCK_NONE",
            "HARM_CATEGORY_DANGEROUS_CONTENT": "BLOCK_NONE"
        }
        # Define um tempo limite para a resposta da API
        request_options = {"timeout": 60} # 60 segundos

        # Chama a API Gemini
        response = modelo_gemini.generate_content(
            prompt_com_comentario,
            safety_settings=safety_settings,
            request_options=request_options
        )

        # Processa a resposta da API
        texto_resposta = response.text.strip()
        sentimento_extraido = "Erro Parsing"
        tema_extraido = "Erro Parsing"
        linhas = texto_resposta.split('\n')

        # Tenta extrair Sentimento e Tema das linhas da resposta
        for linha in linhas:
            linha_strip = linha.strip()
            if linha_strip.lower().startswith("sentimento:"):
                sentimento_extraido = linha_strip.split(":", 1)[1].strip()
            elif linha_strip.lower().startswith("tema:"):
                tema_extraido = linha_strip.split(":", 1)[1].strip()

        # --- Validações e Correções da Resposta ---

        # 1. Verifica se conseguiu extrair ambos
        if sentimento_extraido == "Erro Parsing" or tema_extraido == "Erro Parsing":
            # print(f"Debug - Erro Parsing: '{comentario[:50]}...' -> '{texto_resposta}'") # Log local (opcional)
            return "Erro Parsing", "Erro Parsing"

        # 2. Verifica se o Sentimento é válido
        if sentimento_extraido not in categorias_sentimento_validas:
            # print(f"Debug - Sentimento Inválido: '{sentimento_extraido}' para '{comentario[:50]}...'") # Log local (opcional)
            return "Erro Parsing", "Erro Parsing" # Considera erro de parsing se inválido

        # 3. Regra de Ouro: Se Sentimento é NC, Tema DEVE ser NC(Tema)
        if sentimento_extraido == "Não Classificado":
            # Auto-corrige silenciosamente se a IA retornou tema diferente
            return "Não Classificado", "Não Classificado (Tema)"

        # 4. Se Sentimento é P/N/Neutro, verifica se o Tema é válido E diferente de NC(Tema)
        else: # Sentimento é Positivo, Negativo ou Neutro
            if tema_extraido not in categorias_tema_validas or tema_extraido == "Não Classificado (Tema)":
                 # print(f"Debug - Tema Inválido/NC p/ Sent Válido: '{tema_extraido}' para Sent='{sentimento_extraido}', Msg:'{comentario[:50]}...'") # Log local
                 # Mantém o sentimento válido, mas marca o tema como erro de parsing
                 return sentimento_extraido, "Erro Parsing"
            else:
                 # Tudo OK
                 return sentimento_extraido, tema_extraido

    # --- Tratamento de Erros da API ---
    except genai.types.StopCandidateException as e:
        # Captura especificamente erros de conteúdo bloqueado pelas safety settings
        # print(f"Debug - Erro API (Bloqueado): '{comentario[:50]}...'") # Log local
        return "Erro API", "Erro API (Conteúdo Bloqueado)"
    except Exception as e:
        # Captura outros erros da API (timeout, conexão, etc.)
        error_type = "Erro API (Geral)"
        error_message = str(e).lower()
        if "timeout" in error_message or "deadline exceeded" in error_message:
            error_type = "Erro API (Timeout)"
        # print(f"Debug - {error_type}: '{comentario[:50]}...' Erro: {e}") # Log local
        return "Erro API", error_type

# --- Função para Gerar Insights (NOVA) ---
def gerar_insights(df_resultados_func, modelo_gemini):
    """Analisa o DataFrame de resultados e gera insights usando a API Gemini."""
    if df_resultados_func is None or df_resultados_func.empty:
        return "Não há dados suficientes para gerar insights."
    if not modelo_gemini:
        return "*Erro: Modelo Gemini não inicializado. Não é possível gerar insights.*"

    try:
        total_analisados_func = len(df_resultados_func)

        # 1. Calcular contagens GERAIS de sentimentos (incluindo NC/Erros)
        sent_counts_total = df_resultados_func['Sentimento_Classificado'].value_counts()
        count_pos_func = sent_counts_total.get('Positivo', 0)
        count_neg_func = sent_counts_total.get('Negativo', 0)
        count_neu_func = sent_counts_total.get('Neutro', 0)
        # Soma NC e todos os tipos de erro para o total 'Não Classificado / Erros'
        count_nc_err_func = total_analisados_func - (count_pos_func + count_neg_func + count_neu_func)

        # Calcula percentuais gerais
        perc_pos_func = (count_pos_func / total_analisados_func * 100) if total_analisados_func > 0 else 0
        perc_neg_func = (count_neg_func / total_analisados_func * 100) if total_analisados_func > 0 else 0
        perc_neu_func = (count_neu_func / total_analisados_func * 100) if total_analisados_func > 0 else 0
        perc_nc_err_func = (count_nc_err_func / total_analisados_func * 100) if total_analisados_func > 0 else 0

        # 2. Calcular contagens de temas para Insights (excluindo categorias indesejadas)
        df_tema_insights = df_resultados_func[~df_resultados_func['Tema_Classificado'].isin(categorias_excluir_tema)].copy()
        tema_counts_insights = df_tema_insights['Tema_Classificado'].value_counts()
        top_temas_formatado_func = ""
        total_temas_insights_func = 0
        if not tema_counts_insights.empty:
            total_temas_insights_func = tema_counts_insights.sum()
            # Formata os Top 5 temas para o prompt
            top_temas_formatado_func = "\n".join([
                f"    - {tema}: {count} ({count / total_temas_insights_func * 100:.1f}%)"
                for tema, count in tema_counts_insights.head(5).items() # Pega os Top 5 temas
            ]) if total_temas_insights_func > 0 else "    - Nenhum tema relevante classificado."
        else:
            top_temas_formatado_func = "    - Nenhum tema relevante classificado."

        # 3. Calcular principais temas NEGATIVOS
        df_negativos = df_resultados_func[df_resultados_func['Sentimento_Classificado'] == 'Negativo'].copy()
        # Filtra também os temas negativos excluídos (NC, Erro, Interação)
        df_negativos_filtrados = df_negativos[~df_negativos['Tema_Classificado'].isin(categorias_excluir_tema)]
        tema_neg_counts = df_negativos_filtrados['Tema_Classificado'].value_counts()

        top_temas_negativos_formatado_func = ""
        total_temas_neg_func = 0
        if not tema_neg_counts.empty:
             total_temas_neg_func = tema_neg_counts.sum()
             # Formata os Top 3 temas negativos para o prompt
             top_temas_negativos_formatado_func = "\n".join([
                 f"    - {tema}: {count} ({count / total_temas_neg_func * 100:.1f}%)"
                 for tema, count in tema_neg_counts.head(3).items() # Pega os Top 3 temas negativos
             ]) if total_temas_neg_func > 0 else "    - Nenhum tema negativo relevante classificado."
        else:
             top_temas_negativos_formatado_func = "    - Nenhum tema negativo relevante classificado (ou nenhum comentário negativo com tema válido)."


        # 4. Formatar o prompt final de insights com os dados calculados
        prompt_final_insights = prompt_geracao_insights.format(
            total_comentarios_analisados=total_analisados_func,
            count_pos=count_pos_func, perc_pos=perc_pos_func,
            count_neg=count_neg_func, perc_neg=perc_neg_func,
            count_neu=count_neu_func, perc_neu=perc_neu_func,
            count_nc_err=count_nc_err_func, perc_nc_err=perc_nc_err_func,
            total_temas_insights=total_temas_insights_func, # Adicionado total para contexto
            top_temas_formatado=top_temas_formatado_func,
            total_temas_neg=total_temas_neg_func, # Adicionado total para contexto
            top_temas_negativos_formatado=top_temas_negativos_formatado_func
        )

        # 5. Chamar a API Gemini para gerar os insights
        safety_settings = { "HARM_CATEGORY_HARASSMENT": "BLOCK_NONE", "HARM_CATEGORY_HATE_SPEECH": "BLOCK_NONE",
                           "HARM_CATEGORY_SEXUALLY_EXPLICIT": "BLOCK_NONE", "HARM_CATEGORY_DANGEROUS_CONTENT": "BLOCK_NONE"}
        request_options = {"timeout": 90} # Aumentar um pouco o timeout para a geração de insights
        response_insights = modelo_gemini.generate_content(
            prompt_final_insights,
            safety_settings=safety_settings,
            request_options=request_options
        )

        # 6. Retorna o texto dos insights gerados
        # Verifica se a resposta contém texto antes de tentar acessar .text
        if response_insights and hasattr(response_insights, 'text'):
            st.session_state.insights_generated = response_insights.text.strip() # Guarda no estado
            return st.session_state.insights_generated
        else:
            # Tenta obter informações do erro se a resposta não tiver texto (ex: bloqueio)
            error_info = "Resposta da API vazia ou inválida."
            if response_insights and hasattr(response_insights, 'prompt_feedback'):
                 error_info = f"Possível bloqueio pela API. Feedback: {response_insights.prompt_feedback}"
            st.warning(f"Não foi possível gerar insights: {error_info}", icon="⚠️")
            st.session_state.insights_generated = f"*Não foi possível gerar insights: {error_info}*" # Guarda erro no estado
            return st.session_state.insights_generated


    except Exception as e:
        # Captura erros durante o cálculo ou a chamada da API de insights
        st.error(f"Erro durante a geração de insights: {e}", icon="🚨")
        st.session_state.insights_generated = f"*Ocorreu um erro inesperado durante a geração dos insights: {str(e)}*" # Guarda erro
        return st.session_state.insights_generated


# --- Interface Principal ---
st.title("📊 Aplicativo para análise de sentimento e temática automatizado por IA")
st.markdown("""
Este aplicativo utiliza a IA Generativa do Google (Gemini) para classificar automaticamente o **Sentimento** e a **Temática** de comentários.
Desenvolvido pelo time de Social Intelligence do Hub de Inovação da I&Co. para o Itaú.
""")
st.markdown("---")

# --- Controles na Barra Lateral ---
st.sidebar.divider()
st.sidebar.header("Controles")
uploaded_file = st.sidebar.file_uploader(
    "1. Escolha o arquivo (.csv ou .xlsx)",
    type=["csv", "xlsx"],
    key="file_uploader",
    help="Faça upload de um arquivo CSV ou Excel que contenha uma coluna chamada 'conteúdo' com os textos a serem analisados."
)

# Identifica a coluna de conteúdo (assume 'conteúdo' como padrão)
# Poderia ser um selectbox se quisesse deixar o usuário escolher
coluna_conteudo = 'conteúdo' # Mantenha como 'conteúdo' por enquanto

# Habilita o botão de análise apenas se a API estiver OK e um arquivo tiver sido carregado
botao_habilitado = st.session_state.get('api_key_configured', False) and uploaded_file is not None
analisar_btn = st.sidebar.button(
    "2. Analisar Comentários",
    key="analyze_button",
    disabled=(not botao_habilitado),
    help="Clique para iniciar a análise dos comentários na coluna 'conteúdo' do arquivo carregado."
)

# Mensagens de status na barra lateral
if not st.session_state.get('api_key_configured', False):
    st.sidebar.warning("API Key do Google não configurada ou inválida.", icon="⚠️")
if not uploaded_file:
    st.sidebar.info("Aguardando upload do arquivo...", icon="📤")
if botao_habilitado:
    st.sidebar.info("Pronto para analisar!", icon="✅")


# --- Área Principal: Pré-visualização e Resultados ---
df_original = None
df_para_analise = None
total_comentarios_para_analisar = 0

# Processa o arquivo carregado
if uploaded_file is not None:
    try:
        # Lê o arquivo com base na extensão
        if uploaded_file.name.endswith('.csv'):
            try:
                # Tenta ler como UTF-8 primeiro
                df_original = pd.read_csv(uploaded_file)
            except UnicodeDecodeError:
                # Se falhar, tenta como Latin-1 (comum em exports do Excel)
                uploaded_file.seek(0) # Volta ao início do buffer do arquivo
                df_original = pd.read_csv(uploaded_file, encoding='latin1')
        else: # Assume .xlsx
            df_original = pd.read_excel(uploaded_file)

        # Verifica se a coluna de conteúdo existe
        if coluna_conteudo not in df_original.columns:
            st.error(f"Erro Crítico: Coluna '{coluna_conteudo}' não encontrada no arquivo '{uploaded_file.name}'. Verifique o nome da coluna.", icon="🚨")
            df_original = None # Impede processamento adicional
        else:
            # Prepara o DataFrame para análise: remove linhas com conteúdo vazio/nulo
            df_para_analise = df_original.copy()
            df_para_analise.dropna(subset=[coluna_conteudo], inplace=True)
            df_para_analise = df_para_analise[df_para_analise[coluna_conteudo].astype(str).str.strip() != '']
            total_comentarios_para_analisar = len(df_para_analise)

            # Mostra pré-visualização dos dados originais
            st.subheader("Pré-visualização dos dados originais:")
            st.dataframe(df_original.head(10), use_container_width=True)
            if total_comentarios_para_analisar < len(df_original):
                 st.info(f"Total de linhas no arquivo: {len(df_original)}. Total de comentários válidos para análise (coluna '{coluna_conteudo}' não vazia): **{total_comentarios_para_analisar}**", icon="ℹ️")
            else:
                 st.info(f"Total de comentários válidos para análise: **{total_comentarios_para_analisar}**", icon="ℹ️")

    except Exception as e:
        st.error(f"Erro ao ler ou processar o arquivo '{uploaded_file.name}': {e}", icon="🚨")
        df_original = None
        df_para_analise = None

# Container para os resultados (para poder limpar ou atualizar depois)
results_container = st.container()

# --- Lógica de Análise ---
if analisar_btn and df_para_analise is not None:
    if total_comentarios_para_analisar == 0:
        st.warning("Nenhum comentário válido encontrado na coluna '{coluna_conteudo}' para análise.", icon="⚠️")
    elif not model:
        st.error("Erro: Modelo Gemini não inicializado. Verifique a configuração da API Key na barra lateral.", icon="🚨")
    else:
        # Reseta o estado antes de iniciar nova análise
        st.session_state.analysis_done = False
        st.session_state.df_results = None
        st.session_state.insights_generated = None

        with st.spinner(f"Analisando {total_comentarios_para_analisar} comentários... Isso pode levar alguns minutos."):
            progress_bar = st.progress(0.0) # Inicia a barra de progresso
            status_text = st.empty() # Placeholder para texto de status
            resultados_sentimento = []
            resultados_tema = []
            df_copy_analise = df_para_analise.copy() # Trabalha com cópia

            start_time = time.time() # Medir tempo

            # Itera sobre os comentários válidos para análise
            for i, comentario in enumerate(df_copy_analise[coluna_conteudo]):
                # Chama a função de análise para cada comentário
                sentimento, tema = analisar_comentario(str(comentario), model)
                resultados_sentimento.append(sentimento)
                resultados_tema.append(tema)

                # Atualiza o progresso
                progresso = (i + 1) / total_comentarios_para_analisar
                # Formata o progresso para exibir como percentual
                progress_bar.progress(progresso)
                status_text.text(f"Analisando: {i+1}/{total_comentarios_para_analisar} ({progresso:.1%})")
                # time.sleep(0.1) # Pequena pausa opcional para evitar sobrecarga visual

            end_time = time.time()
            tempo_total = end_time - start_time

            # Limpa elementos de progresso
            progress_bar.empty()
            status_text.success(f"✅ Análise concluída em {tempo_total:.2f} segundos!", icon="🎉")

            # Adiciona os resultados ao DataFrame
            df_copy_analise['Sentimento_Classificado'] = resultados_sentimento
            df_copy_analise['Tema_Classificado'] = resultados_tema

            # Salva os resultados no estado da sessão
            st.session_state.df_results = df_copy_analise
            st.session_state.analysis_done = True
            # st.rerun() # Re-executa o script para exibir os resultados abaixo
            # Não precisa de rerun aqui, pois os resultados serão exibidos na mesma execução

# --- Exibição dos Resultados ---
if st.session_state.analysis_done and st.session_state.df_results is not None:
    with results_container:
        df_results = st.session_state.df_results
        total_analisados_results = len(df_results)

        st.markdown("---")
        st.subheader("Visualização dos Resultados")

        # --- Cálculo para NPS e Gráficos ---
        # Filtra apenas sentimentos válidos para gráficos (P, N, Neu)
        df_sent_chart = df_results[~df_results['Sentimento_Classificado'].isin(categorias_excluir_sentimento)].copy()
        sent_counts_chart = df_sent_chart['Sentimento_Classificado'].value_counts()
        total_sent_chart = sent_counts_chart.sum()
        nps_score_num = None

        # Calcula NPS se houver dados válidos
        if total_sent_chart > 0:
            count_pos_chart = sent_counts_chart.get('Positivo', 0)
            count_neu_chart = sent_counts_chart.get('Neutro', 0)
            count_neg_chart = sent_counts_chart.get('Negativo', 0)
            # Fórmula NPS adaptada (0-10): Promotores(P) - Detratores(N) + (Neutros * 0.5) normalizado
            # Basicamente, (Pos% - Neg%) * 5 + 5, com neutros valendo metade de positivo
            perc_pos_chart = count_pos_chart / total_sent_chart
            perc_neg_chart = count_neg_chart / total_sent_chart
            # O cálculo original estava um pouco diferente, ajustando para escala 0-10 onde P=10, N=0, Neu=5
            # NPS Score = (Promoter % - Detractor %) * 100 (escala -100 a 100)
            # Adaptando para 0-10: ((P% - N%) + 1) * 5 parece mais padrão
            nps_formula_standard = ((perc_pos_chart - perc_neg_chart) + 1) / 2 * 10 # Mapeia [-1, 1] para [0, 10]
            nps_score_num = max(0, min(10, nps_formula_standard)) # Garante que fique entre 0 e 10


        # --- Exibição do NPS e Gráficos ---
        nps_col, chart_col1, chart_col2 = st.columns([1, 2, 2]) # Ajuste de proporção das colunas
        with nps_col:
            st.markdown("##### NPS Social")
            if nps_score_num is not None:
                st.metric(label="(Escala 0-10)", value=f"{nps_score_num:.1f}")
            else:
                st.metric(label="(Escala 0-10)", value="N/A")
                st.caption("Sem dados P/N/Neu.")

        with chart_col1:
            st.markdown("##### Distribuição de Sentimento")
            if total_sent_chart > 0:
                df_plot_sent = pd.DataFrame({'Sentimento': sent_counts_chart.index, 'Volume': sent_counts_chart.values})
                # Ordenar para consistência visual (opcional)
                df_plot_sent['Sentimento'] = pd.Categorical(df_plot_sent['Sentimento'], categories=["Positivo", "Neutro", "Negativo"], ordered=True)
                df_plot_sent = df_plot_sent.sort_values('Sentimento')

                fig_sent = px.pie(df_plot_sent, names='Sentimento', values='Volume',
                                  hole=0.4, # Gráfico de rosca
                                  color='Sentimento',
                                  color_discrete_map={'Positivo': '#28a745', 'Negativo': '#dc3545', 'Neutro': '#ffc107'},
                                  title='Sentimentos (Excluindo Não Classif./Erros)')
                fig_sent.update_traces(textposition='outside', textinfo='percent+label',
                                       hovertemplate="<b>%{label}</b><br>Volume: %{value}<br>Percentual: %{percent:.1%}<extra></extra>")
                fig_sent.update_layout(showlegend=False, title_x=0.5, height=350, margin=dict(l=10, r=10, t=40, b=10))
                st.plotly_chart(fig_sent, use_container_width=True)
            else:
                st.warning("Nenhum sentimento Positivo, Negativo ou Neutro classificado para exibir gráfico.", icon="📊")

        with chart_col2:
            st.markdown("##### Distribuição Temática")
            # Filtra temas indesejados para o gráfico principal
            df_tema_chart = df_results[~df_results['Tema_Classificado'].isin(categorias_excluir_tema)].copy()
            tema_counts_chart = df_tema_chart['Tema_Classificado'].value_counts()
            total_tema_chart = tema_counts_chart.sum()

            if total_tema_chart > 0:
                tema_perc_chart = (tema_counts_chart / total_tema_chart * 100)
                df_plot_tema = pd.DataFrame({
                    'Tema': tema_counts_chart.index,
                    'Volume': tema_counts_chart.values,
                    'Percentual': tema_perc_chart.values
                }).sort_values(by='Volume', ascending=False) # Ordena por volume

                fig_tema = px.bar(df_plot_tema, x='Tema', y='Volume',
                                  color_discrete_sequence=['#007bff']*len(df_plot_tema), # Azul padrão
                                  title='Principais Temas (Excluindo NC/Erro/Interação)',
                                  # Texto no hover mostra Volume e Percentual
                                  hover_data={'Tema': False, 'Volume': True, 'Percentual': ':.1f%'},
                                  text='Volume' # Mostra o volume na barra
                                 )
                fig_tema.update_traces(textposition='outside')
                fig_tema.update_layout(xaxis_title=None, yaxis_title="Volume Bruto", title_x=0.5, height=350,
                                       margin=dict(l=10, r=10, t=40, b=10))
                fig_tema.update_xaxes(tickangle= -30) # Ajusta ângulo dos labels do eixo X
                st.plotly_chart(fig_tema, use_container_width=True)
            else:
                st.warning("Nenhum tema válido (excluindo NC/Erro/Interação) classificado para exibir gráfico.", icon="📊")


        # --- Tabelas de Resumo (Completas, incluindo Erros e NC) ---
        st.markdown("---")
        st.subheader("Tabelas de Resumo Completas")
        col_t1, col_t2 = st.columns(2)

        with col_t1:
            st.markdown("###### Tabela 1: Sentimento (Completa)")
            # Reindexa para garantir que todas as categorias (válidas + erros) apareçam, mesmo se contagem for 0
            todas_cats_sent = categorias_sentimento_validas + sorted(list(set(todas_categorias_erro))) # Ordena erros
            sent_counts_all = df_results['Sentimento_Classificado'].value_counts().reindex(todas_cats_sent, fill_value=0)
            sent_perc_all = (sent_counts_all / total_analisados_results * 100) if total_analisados_results > 0 else 0
            tabela_sent = pd.DataFrame({
                'Sentimento': sent_counts_all.index,
                'Volume Bruto': sent_counts_all.values,
                'Percentual (%)': sent_perc_all.values
            })
            # Adiciona linha Total
            total_row_sent = pd.DataFrame({
                'Sentimento': ['Total Geral'],
                'Volume Bruto': [total_analisados_results],
                'Percentual (%)': [100.0]
            })
            # Mostra apenas categorias com volume > 0, mais a linha Total
            tabela_sent_final = pd.concat([tabela_sent[tabela_sent['Volume Bruto'] > 0], total_row_sent], ignore_index=True)
            st.table(tabela_sent_final.style.format({'Percentual (%)': '{:.2f}%'}))

        with col_t2:
            st.markdown("###### Tabela 2: Temática (Completa)")
            # Reindexa para garantir todas as categorias de tema (válidas + erros)
            todas_cats_tema = categorias_tema_validas + sorted(list(set(todas_categorias_erro))) # Ordena erros
            tema_counts_all = df_results['Tema_Classificado'].value_counts().reindex(todas_cats_tema, fill_value=0)
            # Remove duplicatas no índice se houver sobreposição (embora não devesse com a lógica atual)
            tema_counts_all = tema_counts_all[~tema_counts_all.index.duplicated(keep='first')]
            tema_perc_all = (tema_counts_all / total_analisados_results * 100) if total_analisados_results > 0 else 0
            tabela_tema = pd.DataFrame({
                'Tema': tema_counts_all.index,
                'Volume Bruto': tema_counts_all.values,
                'Percentual (%)': tema_perc_all.values
            })
            # Adiciona linha Total
            total_row_tema = pd.DataFrame({
                'Tema': ['Total Geral'],
                'Volume Bruto': [total_analisados_results],
                'Percentual (%)': [100.0]
            })
            # Mostra apenas categorias com volume > 0, mais a linha Total
            tabela_tema_final = pd.concat([tabela_tema[tabela_tema['Volume Bruto'] > 0], total_row_tema], ignore_index=True)
            st.table(tabela_tema_final.style.format({'Percentual (%)': '{:.2f}%'}))


        # --- Tabela Completa e Download ---
        st.markdown("---")
        st.subheader("Resultados Completos Detalhados")
        # Mostra o dataframe com as novas colunas de classificação
        st.dataframe(df_results, use_container_width=True)

        # Função para converter DataFrame para CSV (cache para performance)
        @st.cache_data # Cacheia o resultado da conversão
        def convert_df_to_csv(df_conv):
            return df_conv.to_csv(index=False, encoding='utf-8-sig').encode('utf-8-sig') # utf-8-sig para Excel ler acentos corretamente

        # Prepara o nome do arquivo de download
        if uploaded_file:
            base_name = uploaded_file.name.split('.')[0]
            download_filename = f"{base_name}_analise_gemini.csv"
        else:
            download_filename = 'analise_gemini_resultados.csv'

        # Gera o CSV para download
        csv_output = convert_df_to_csv(df_results)

        # Botão de Download
        st.download_button(
            label="💾 Download Resultados Completos (.csv)",
            data=csv_output,
            file_name=download_filename,
            mime='text/csv',
            key='download_csv',
            help="Baixa a tabela completa acima, incluindo as classificações de Sentimento e Tema, em formato CSV."
        )

        # --- Seção de Geração e Exibição de Insights (NOVA) ---
        st.markdown("---")
        st.subheader("💡 Insights e Percepções Acionáveis")

        # Verifica se a análise foi feita, temos resultados, modelo E se insights ainda não foram gerados/carregados
        if st.session_state.analysis_done and st.session_state.df_results is not None and model:
            # Verifica se os insights já foram gerados nesta sessão para evitar re-gerar a cada interação
            if st.session_state.insights_generated is None:
                with st.spinner("Gerando insights com base nos resultados..."):
                     # Chama a função para gerar insights (ela já salva no session_state)
                     gerar_insights(st.session_state.df_results, model)

            # Exibe os insights que estão salvos no estado da sessão
            if st.session_state.insights_generated:
                 st.markdown(st.session_state.insights_generated) # Exibe os insights formatados em Markdown
            else:
                 st.warning("Não foi possível gerar ou carregar os insights.", icon="⚠️")

        elif not model:
             st.warning("Modelo Gemini não inicializado. Não é possível gerar insights.", icon="⚠️")
        else:
             st.info("Realize uma análise primeiro para poder gerar os insights.", icon="ℹ️")


# Mensagem inicial se nenhum arquivo foi carregado ainda
elif not uploaded_file and not st.session_state.analysis_done :
     st.info("⬅️ Para começar, configure sua API Key (se necessário) e faça o upload de um arquivo .csv ou .xlsx na barra lateral.", icon="👈")