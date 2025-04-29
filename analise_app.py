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

# --- Configuração da API Key ---
api_key_source = None
try:
    if hasattr(st, 'secrets') and "GOOGLE_API_KEY" in st.secrets:
        st.session_state.api_key_input_value = st.secrets["GOOGLE_API_KEY"]
        api_key_source = "secrets"
except Exception as e: st.sidebar.warning(f"Não foi possível ler os secrets: {e}")

# --- Interface da Barra Lateral para API Key ---
st.sidebar.header("Configuração")
if api_key_source != "secrets":
    user_provided_key = st.sidebar.text_input(
        "Insira sua Google API Key aqui:", type="password",
        key="api_key_widget", value=st.session_state.api_key_input_value
    )
    if user_provided_key != st.session_state.api_key_input_value:
         st.session_state.api_key_input_value = user_provided_key; st.session_state.api_key_configured = False
else:
    st.sidebar.success("API Key carregada dos segredos!", icon="✅")
    if not st.session_state.api_key_configured: st.session_state.api_key_configured = False

# --- Tentativa de Configurar a API e o Modelo ---
model = None
if st.session_state.api_key_input_value and not st.session_state.api_key_configured:
    try:
        genai.configure(api_key=st.session_state.api_key_input_value)
        model = genai.GenerativeModel('gemini-1.5-flash')
        st.session_state.api_key_configured = True
        if api_key_source != "secrets": st.sidebar.success("API Key configurada!", icon="🔑")
        st.sidebar.caption(f"Modelo Gemini: gemini-1.5-flash")
    except Exception as e: st.sidebar.error(f"Erro API Key/Modelo. Verifique."); st.session_state.api_key_configured = False; model = None
elif st.session_state.api_key_configured:
     try: model = genai.GenerativeModel('gemini-1.5-flash')
     except Exception as e: st.sidebar.error(f"Erro Modelo: {e}"); st.session_state.api_key_configured = False; model = None

# --- Prompt REFINADO v3 (Ajuste em Menção Isolada) ---
seu_prompt_completo = """
Persona: Você é uma IA Analista de Feedback de Clientes e Social Listening de um banco Brasileiro, otimizada para classificar mensagens em Português do Brasil (Pt-BR) com alta precisão contextual.
Objetivo: Classificar CADA mensagem em UMA categoria de Sentimento (Positivo, Negativo, Neutro, Não Classificado) e UMA categoria Temática. Aderir estritamente às definições e regras. Minimizar 'Não Classificado'.
Contexto Geral: Mensagens de clientes/público sobre posts do banco (Itaú e submarcas) e suas iniciativas (produtos, serviços, campanhas, eventos, patrocínios). Assuma que reações curtas são contextuais ao post original.

=== REGRAS GERAIS E DE OURO ===
1.  Análise Dupla Obrigatória: Sentimento + Tema.
2.  Priorize P/N/Neutro: Só use 'Não Classificado' como ÚLTIMO recurso absoluto, conforme definições estritas abaixo.
3.  Vinculação NC: Se Sentimento = Não Classificado, então Tema = Não Classificado (Tema). SEMPRE.
4.  **REGRA CRÍTICA - Menção Isolada:** Uma mensagem contendo APENAS uma menção de usuário (ex: "@nomeusuario", "[nome usuario]") SEM NENHUM outro texto, número ou emoji, DEVE SER CLASSIFICADA COMO: Sentimento: Positivo, Tema: Interação Social. Isso representa compartilhamento/engajamento.
5.  Formato de Resposta: EXATAMENTE DUAS LINHAS:
    Sentimento: [Nome Exato da Categoria de Sentimento]
    Tema: [Nome Exato da Categoria de Tema]
    (Não inclua NADA MAIS, nem explicações, nem markdown).

=== DEFINIÇÕES DE SENTIMENTO (Escolha UMA) ===

**Positivo:** Expressa satisfação, apoio, entusiasmo, gratidão, apreciação, concordância.
    *   Indicadores: Elogios ("Amei", "Top", "Ótimo banco"), agradecimentos ("Obg"), apoio ("Parabéns", "Torcendo"), apreciação ("Belo post"); Emojis positivos isolados ou dominantes (😍, ❤️, 👍, 🎉, ✨, 👏, 🙏-gratidão, 😉); **Menções ESTRITAMENTE ISOLADAS (@usuario, [nome]) - Ver Regra Crítica 4 acima**; Textos curtos de concordância ("Isso", "Exato"). Ênfase (!!!!) pode ser Positiva se o tom geral for de excitação/apoio.
    *   Ex: "Parabéns pela iniciativa!!!!", "@itau <0xF0><0x9F><0x91><0x8F>", "[Luiz Erik]" (como mensagem única), "😉"

**Negativo:** Expressa insatisfação, crítica, raiva, frustração, reclamação, tristeza, relato de problema/golpe/erro.
    *   Indicadores: Críticas ("Péssimo", "Lixo", "Decepção"), reclamações ("Atendimento horrível", "Não resolvem"), relatos de problemas ("App não funciona", "Cobrança indevida", "Fui roubado", "Não consigo acesso"), insatisfação ("Taxa alta", "Demora"), frustração (CAIXA ALTA com teor negativo, "!!!!" após crítica), advertência ("Não recomendo"); Emojis negativos isolados ou dominantes (😠, 😡, 👎, 😢, 🤮, 💩). Qualquer menção a golpe, fraude, roubo, erro grave é Negativa.
    *   Ex: "Péssimo atendimento!!!!", "Não consigo usar o app.", "Fui vítima de golpe @itau", "🤮"

**Neutro:** Busca/fornece informação, observação factual, pergunta, sugestão objetiva, reação sem forte valência P/N.
    *   Indicadores: Perguntas ("Como faço?", "Qual o telefone?", "Quando começa?"), pedidos ("Me ajuda", "Gostaria de..."), informações factuais ("O evento é amanhã"), sugestões ("Poderiam fazer X"), observações ("Entendi", "Ok"); Emojis neutros isolados ou dominantes (🤔, 👀, 😂, 😅, 🙏-"por favor"); **Múltiplas menções isoladas** (@itau @outro). Frases que iniciam pergunta/pedido após menção (@itau Qual o horário?). Termos/siglas específicos sobre o tópico sem juízo de valor ("Faz o L", "ESG").
    *   Ex: "@itau Qual o endereço?", "Acho que é em SP", "Ok", "@itau @cliente Olá", "Faz o L"

**Não Classificado:** APLICAR SOMENTE E EXCLUSIVAMENTE SE (e NUNCA para menção estritamente isolada):
    1.  **Idioma Estrangeiro:** Predominantemente não em Pt-BR (Ex: "What time?", "@itau guess I").
    2.  **Incompreensível:** Erros graves, digitação aleatória, texto sem sentido lógico (Ex: "asdf ghjk", "..oitoitoitame"). Inclui letras/números isolados sem contexto claro APÓS uma menção ou isolados (Ex: "@itau p", "@2").
    3.  **Spam/Link Isolado:** Conteúdo repetitivo óbvio, promoções não relacionadas, URL isolada SEM contexto relevante (Ex: "Confira: https://...", "https://t.co/xxxxx").
    4.  **Totalmente Off-Topic:** Assunto sem QUALQUER relação com banco, finanças, campanha, evento, etc. (Ex: "Receita de bolo").
    5.  **Interação Social Pura ISOLADA (Texto/Emoji):** APENAS saudações textuais ("Bom dia"), APENAS risadas textuais ("kkkk"), APENAS emojis que não se encaixam claramente em P/N/Neutro pelo contexto OU que não são menções. (Se tiver "kkkk adorei", classifique "adorei").
    *   Ex: "Bom dia", "rsrsrs", "https://example.com", "jsjsjshshs", "<>", "**"

=== DEFINIÇÕES DE TEMA (Escolha UMA, vinculada ao Sentimento) ===

***IMPORTANTE: Use EXATAMENTE um dos nomes de Tema 1 a 9 abaixo. Se Sentimento = Não Classificado, Tema = Não Classificado (Tema).***

1.  **Elogio Geral (Marca/Evento/Conteúdo/Experiência):** Elogios sobre a marca, posts, eventos, experiência geral. (Sentimento: Positivo)
2.  **Elogio Específico (Pessoa/Figura Pública/Representante/"Laranjinha"):** Elogios a pessoas específicas (Jorge Ben Jor, atletas, "laranjinhas"). (Sentimento: Positivo)
3.  **Reclamação/Crítica (Serviços/Produtos/Atendimento/Políticas):** Reclamações sobre atendimento, produtos (cartão, conta), taxas, políticas, golpes, fraudes, segurança. (Sentimento: Negativo)
4.  **Problemas Técnicos (Plataformas/Funcionalidades):** Relatos de falhas em app, site, caixa eletrônico, Pix, reconhecimento facial, etc. (Sentimento: Negativo)
5.  **Apoio/Incentivo (Pessoas/Causas/Marca):** Mensagens de apoio, torcida para pessoas, causas ou a marca/iniciativa. (Sentimento: Positivo)
6.  **Solicitação/Dúvida/Sugestão:** Perguntas, pedidos de informação/ajuda, sugestões sobre serviços, produtos, eventos, etc. (Sentimento: Neutro)
7.  **Interação Social:** Usado para: **Menções estritamente isoladas (@ ou []) classificadas como Positivo**; Múltiplas menções isoladas (classificadas como Neutro); Emojis isolados P/N/Neutro sem contexto temático forte; Interações curtas P/N/Neutro focadas na menção (@itau vc). (Sentimento: Positivo, Negativo ou Neutro, NUNCA Não Classificado).
8.  **Discussão Específica (Tópico da Campanha/Evento):** Comentários sobre o tema central do post/campanha/evento (música, jogo, detalhe do evento, pessoa em destaque), incluindo observações factuais ou termos específicos ("Faz o L"). (Sentimento: Pode ser Positivo, Negativo ou Neutro)
9.  **Não Classificado (Tema):** Exclusivamente quando Sentimento = Não Classificado. Engloba idioma estrangeiro, incompreensível, spam, off-topic, interação social pura isolada (texto/emoji, não menção). (Sentimento: Não Classificado)

=== REGRAS ADICIONAIS DE CLASSIFICAÇÃO ===
*   **Menções + Texto:** O sentimento/tema deve seguir o TEXTO. Se o texto for claro (elogio, crítica, pergunta), use o sentimento/tema do texto. Se o texto for ambíguo/curto (ex: "@itau vc", "@itau aqui") que não se encaixe em P/N/Neutro claros, classifique como **Positivo / Interação Social** (priorizando a intenção de marcar).
*   **Emojis Mistos:** Prioridade: Negativo > Positivo > Neutro. (Ex: 🤔❤️👎 -> Negativo; 🤔❤️👍 -> Positivo; 🤔👀 -> Neutro).
*   **Ênfase (!!!, ???):** Modifica o sentimento base, não o define isoladamente. "Ótimo!!!" -> Positivo. "Péssimo!!!" -> Negativo. "!!!!" isolado pode ser Positivo (excitação) dependendo do contexto implícito do post.
*   **Mensagens Mistas:** Classifique pelo elemento PREDOMINANTE ou mais FORTE (Reclamação > outros; Pergunta > outros).

Agora, analise a seguinte mensagem:
{comment}
"""

# --- Listas de Categorias Válidas ---
# (Mantidas como antes)
categorias_sentimento_validas = ["Positivo", "Negativo", "Neutro", "Não Classificado"]
categorias_tema_validas = [
    "Elogio Geral (Marca/Evento/Conteúdo/Experiência)", "Elogio Específico (Pessoa/Figura Pública/Representante/\"Laranjinha\")",
    "Reclamação/Crítica (Serviços/Produtos/Atendimento/Políticas)", "Problemas Técnicos (Plataformas/Funcionalidades)",
    "Apoio/Incentivo (Pessoas/Causas/Marca)", "Solicitação/Dúvida/Sugestão", "Interação Social",
    "Discussão Específica (Tópico da Campanha/Evento)", "Não Classificado (Tema)"
]
categorias_erro = ["Erro Parsing", "Erro API"]
categorias_erro_tema_especifico = ["Erro API (Timeout)", "Erro API (Geral)", "Erro API (Modelo não iniciado)", "Erro API (Conteúdo Bloqueado)"] # Adicionado bloqueio
todas_categorias_erro = list(set(categorias_erro + categorias_erro_tema_especifico))
categorias_excluir_sentimento = ["Não Classificado"] + todas_categorias_erro
categorias_excluir_tema = ["Não Classificado (Tema)"] + todas_categorias_erro


# --- Função para Analisar um Comentário (Lógica interna revisada ligeiramente para erros) ---
def analisar_comentario(comentario, modelo_gemini):
    """Envia um comentário para a API Gemini e retorna o sentimento e tema classificados."""
    if not comentario or not isinstance(comentario, str) or comentario.strip() == "": return "Não Classificado", "Não Classificado (Tema)"
    if not modelo_gemini: return "Erro API", "Erro API (Modelo não iniciado)"

    prompt_com_comentario = seu_prompt_completo.format(comment=comentario)
    try:
        safety_settings = { "HARM_CATEGORY_HARASSMENT": "BLOCK_NONE", "HARM_CATEGORY_HATE_SPEECH": "BLOCK_NONE",
                           "HARM_CATEGORY_SEXUALLY_EXPLICIT": "BLOCK_NONE", "HARM_CATEGORY_DANGEROUS_CONTENT": "BLOCK_NONE"}
        request_options = {"timeout": 60}
        response = modelo_gemini.generate_content(prompt_com_comentario, safety_settings=safety_settings, request_options=request_options)

        texto_resposta = response.text.strip(); sentimento_extraido = "Erro Parsing"; tema_extraido = "Erro Parsing"
        linhas = texto_resposta.split('\n')
        for linha in linhas:
            linha_strip = linha.strip()
            if linha_strip.lower().startswith("sentimento:"): sentimento_extraido = linha_strip.split(":", 1)[1].strip()
            elif linha_strip.lower().startswith("tema:"): tema_extraido = linha_strip.split(":", 1)[1].strip()

        if sentimento_extraido == "Erro Parsing" or tema_extraido == "Erro Parsing":
             # Não mostra warning para cada erro de parsing, pode poluir muito. Log interno talvez.
             # st.warning(f"Formato resp. inesperado '{comentario[:50]}...': '{texto_resposta}'", icon="⚠️")
             return "Erro Parsing", "Erro Parsing"
        if sentimento_extraido not in categorias_sentimento_validas:
             # st.warning(f"Sent. inválido '{sentimento_extraido}' retornado: '{comentario[:50]}...'", icon="⚠️")
             return "Erro Parsing", "Erro Parsing"

        if sentimento_extraido == "Não Classificado":
            # Se IA retornar tema diferente de NC, auto-corrige silenciosamente.
            return "Não Classificado", "Não Classificado (Tema)"
        else: # Sentimento P/N/Neutro
             if tema_extraido not in categorias_tema_validas or tema_extraido == "Não Classificado (Tema)":
                  # st.warning(f"Tema inválido '{tema_extraido}' p/ Sent='{sentimento_extraido}'. Msg:'{comentario[:50]}...'", icon="⚠️")
                  return sentimento_extraido, "Erro Parsing" # Mantem sentimento, marca tema como erro
             else:
                  return sentimento_extraido, tema_extraido # OK

    except genai.types.StopCandidateException as e:
        # Captura especificamente erros de conteúdo bloqueado pela safety settings
        # st.error(f"Erro API (Conteúdo Bloqueado): '{comentario[:50]}...'.", icon="🚨") # Log silencioso talvez?
        return "Erro API", "Erro API (Conteúdo Bloqueado)"
    except Exception as e:
        error_type = "Erro API (Geral)"
        if "timeout" in str(e).lower() or "Deadline exceeded" in str(e): error_type = "Erro API (Timeout)"
        # st.error(f"{error_type}: '{comentario[:50]}...'.", icon="🚨") # Log silencioso talvez?
        return "Erro API", error_type

# --- Interface Principal (Layout e Lógica de Exibição sem mudanças) ---
# ... (O restante do código continua igual ao da versão anterior,
#      incluindo a exibição do título, subtítulo, pré-visualização,
#      lógica de análise em lote, cálculo de NPS, gráficos e tabelas)
# ... (Copie e cole todo o restante do código da resposta anterior aqui)

# --- Copie o restante do código da Interface Principal daqui para baixo ---
st.title("📊 Aplicativo para análise de sentimento e temática automatizado por IA")
st.markdown("Este aplicativo foi desenvolvido pelo time de Social Intelligence do Hub de Inovação da Ihouse para o Itaú. As análises são realizadas e geradas através do Gemini.")
st.markdown("---")

# --- Controles na Barra Lateral ---
st.sidebar.divider()
st.sidebar.header("Controles")
uploaded_file = st.sidebar.file_uploader("1. Escolha o arquivo (.csv ou .xlsx)", type=["csv", "xlsx"], key="file_uploader")
botao_habilitado = st.session_state.get('api_key_configured', False) and uploaded_file is not None
analisar_btn = st.sidebar.button("2. Analisar Comentários", key="analyze_button", disabled=(not botao_habilitado))
if not st.session_state.get('api_key_configured', False): st.sidebar.warning("API Key não configurada.")
if not uploaded_file: st.sidebar.info("Aguardando upload do arquivo...")

# --- Área Principal: Pré-visualização e Resultados ---
df = None; total_comentarios_validos = 0
if uploaded_file is not None:
    try:
        if uploaded_file.name.endswith('.csv'):
            try: df_original = pd.read_csv(uploaded_file)
            except UnicodeDecodeError: uploaded_file.seek(0); df_original = pd.read_csv(uploaded_file, encoding='latin1')
        else: df_original = pd.read_excel(uploaded_file)
        df = df_original.copy()
        if 'conteúdo' not in df.columns: st.error("Erro Crítico: Coluna 'conteúdo' não encontrada."); df = None
        else:
            df.dropna(subset=['conteúdo'], inplace=True); df = df[df['conteúdo'].astype(str).str.strip() != '']
            total_comentarios_validos = len(df)
    except Exception as e: st.error(f"Erro ao ler o arquivo: {e}"); df = None

if df is not None:
    st.subheader("Pré-visualização dos dados:")
    st.dataframe(df.head(10), use_container_width=True)
    st.info(f"Total de comentários válidos (não vazios) encontrados: **{total_comentarios_validos}**")

    results_container = st.container()
    if analisar_btn:
        if total_comentarios_validos == 0: st.warning("Nenhum comentário válido para análise.")
        elif not model: st.error("Erro: Modelo Gemini não inicializado. Verifique a API Key.")
        else:
            with st.spinner(f"Analisando {total_comentarios_validos} comentários... Isso pode levar alguns minutos."):
                progress_bar = st.progress(0); status_text = st.empty()
                resultados_sentimento = []; resultados_tema = []
                df_copy = df.copy()
                for i, comentario in enumerate(df_copy['conteúdo']):
                    sentimento, tema = analisar_comentario(str(comentario), model)
                    resultados_sentimento.append(sentimento); resultados_tema.append(tema)
                    progresso = (i + 1) / total_comentarios_validos
                    progress_bar.progress(progresso); status_text.text(f"Analisando: {i+1}/{total_comentarios_validos}")
                progress_bar.empty(); status_text.success(f"✅ Análise concluída!")
                df_copy['Sentimento_Classificado'] = resultados_sentimento; df_copy['Tema_Classificado'] = resultados_tema
                st.session_state.df_results = df_copy; st.session_state.analysis_done = True
                st.rerun()

    if st.session_state.analysis_done and st.session_state.df_results is not None:
        df_results = st.session_state.df_results
        total_analisados = len(df_results)

        with results_container:
            st.markdown("---")
            st.subheader("Visualização dos Resultados")

            # --- Cálculo para NPS e Gráficos ---
            df_sent_chart = df_results[~df_results['Sentimento_Classificado'].isin(categorias_excluir_sentimento)].copy()
            sent_counts_chart = df_sent_chart['Sentimento_Classificado'].value_counts()
            total_sent_chart = sent_counts_chart.sum(); nps_score_num = None
            if total_sent_chart > 0:
                count_pos = sent_counts_chart.get('Positivo', 0); count_neu = sent_counts_chart.get('Neutro', 0); count_neg = sent_counts_chart.get('Negativo', 0)
                perc_pos = count_pos / total_sent_chart; perc_neu = count_neu / total_sent_chart; perc_neg = count_neg / total_sent_chart
                nps_score_num = max(0, min(10, ((perc_pos + (perc_neu * 0.5) - perc_neg) * 5) + 5))

            # --- Exibição do NPS e Gráficos ---
            nps_col, chart_col1, chart_col2 = st.columns([1, 2, 2])
            with nps_col:
                st.markdown("##### NPS Social");
                if nps_score_num is not None: st.metric(label="(Escala 0-10)", value=f"{nps_score_num:.1f}")
                else: st.metric(label="(Escala 0-10)", value="N/A"); st.caption("Sem dados P/N/Neu.")
            with chart_col1:
                st.markdown("##### Distribuição de Sentimento")
                if total_sent_chart > 0:
                    df_plot_sent = pd.DataFrame({'Sentimento': sent_counts_chart.index, 'Volume': sent_counts_chart.values})
                    fig_sent = px.pie(df_plot_sent, names='Sentimento', values='Volume', hole=0.4, color='Sentimento', color_discrete_map={'Positivo': '#28a745', 'Negativo': '#dc3545', 'Neutro': '#ffc107'}, title='Sentimentos (Excluindo Não Classif.)')
                    fig_sent.update_traces(textposition='inside', textinfo='percent+label', hovertemplate="<b>%{label}</b><br>Vol: %{value}<br>%: %{percent:.1%}<extra></extra>")
                    fig_sent.update_layout(showlegend=False, title_x=0.5, height=350, margin=dict(l=10, r=10, t=40, b=10)); st.plotly_chart(fig_sent, use_container_width=True)
                else: st.warning("Nenhum sentimento P/N/Neu classificado.")
            with chart_col2:
                st.markdown("##### Distribuição Temática")
                df_tema_chart = df_results[~df_results['Tema_Classificado'].isin(categorias_excluir_tema)]
                tema_counts_chart = df_tema_chart['Tema_Classificado'].value_counts()
                total_tema_chart = tema_counts_chart.sum()
                if total_tema_chart > 0:
                    tema_perc_chart = (tema_counts_chart / total_tema_chart * 100)
                    df_plot_tema = pd.DataFrame({'Tema': tema_counts_chart.index, 'Volume': tema_counts_chart.values, 'Percentual': tema_perc_chart.values}).sort_values(by='Volume', ascending=False)
                    fig_tema = px.bar(df_plot_tema, x='Tema', y='Volume', color_discrete_sequence=['#FFA500']*len(df_plot_tema), title='Temas (Excluindo Não Classif.)', text=df_plot_tema.apply(lambda row: f"{row['Volume']}<br>({row['Percentual']:.1f}%)", axis=1))
                    fig_tema.update_traces(textposition='outside'); fig_tema.update_layout(xaxis_title=None, yaxis_title="Volume Bruto", title_x=0.5, height=350, margin=dict(l=10, r=10, t=40, b=10))
                    fig_tema.update_xaxes(tickangle= -45); st.plotly_chart(fig_tema, use_container_width=True)
                else: st.warning("Nenhum tema válido classificado.")

            # --- Tabelas de Resumo ---
            st.markdown("---"); st.subheader("Tabelas de Resumo")
            col_t1, col_t2 = st.columns(2)
            with col_t1:
                st.markdown("###### Tabela 1: Sentimento (Completa)")
                # Inclui Erro API (Conteúdo Bloqueado) na lista de erros
                todas_cats_sent = categorias_sentimento_validas + todas_categorias_erro
                sent_counts = df_results['Sentimento_Classificado'].value_counts().reindex(todas_cats_sent, fill_value=0)
                sent_perc = (sent_counts / total_analisados * 100).round(2) if total_analisados > 0 else 0
                tabela_sent = pd.DataFrame({'Sentimento': sent_counts.index, 'Volume Bruto': sent_counts.values, 'Percentual (%)': sent_perc.values})
                total_sent = pd.DataFrame({'Sentimento': ['Total'], 'Volume Bruto': [total_analisados], 'Percentual (%)': [100.0]})
                tabela_sent = pd.concat([tabela_sent[tabela_sent['Volume Bruto'] > 0], total_sent], ignore_index=True)
                st.table(tabela_sent.style.format({'Percentual (%)': '{:.2f}%'}))
            with col_t2:
                st.markdown("###### Tabela 2: Temática (Completa)")
                todas_cats_tema = categorias_tema_validas + todas_categorias_erro
                tema_counts = df_results['Tema_Classificado'].value_counts().reindex(todas_cats_tema, fill_value=0)
                tema_counts = tema_counts[~tema_counts.index.duplicated(keep='first')]
                tema_perc = (tema_counts / total_analisados * 100).round(2) if total_analisados > 0 else 0
                tabela_tema = pd.DataFrame({'Tema': tema_counts.index, 'Volume Bruto': tema_counts.values, 'Percentual (%)': tema_perc.values})
                total_tema = pd.DataFrame({'Tema': ['Total'], 'Volume Bruto': [total_analisados], 'Percentual (%)': [100.0]})
                tabela_tema = pd.concat([tabela_tema[tabela_tema['Volume Bruto'] > 0], total_tema], ignore_index=True)
                st.table(tabela_tema.style.format({'Percentual (%)': '{:.2f}%'}))

            # --- Tabela Completa e Download ---
            st.markdown("---"); st.subheader("Resultados Completos Detalhados")
            st.dataframe(df_results, use_container_width=True)
            @st.cache_data
            def convert_df_to_csv(df_conv): return df_conv.to_csv(index=False).encode('utf-8-sig')
            csv_output = convert_df_to_csv(df_results)
            st.download_button("💾 Download Resultados (.csv)", csv_output, 'analise_gemini_resultados.csv', 'text/csv', key='download_csv')

elif not uploaded_file and not analisar_btn :
     st.info("⬅️ Faça o upload de um arquivo .csv ou .xlsx na barra lateral para começar.")