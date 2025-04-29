import streamlit as st
import pandas as pd
import google.generativeai as genai
import io # Para lidar com o arquivo em memória
import time # Para possíveis pausas
import plotly.express as px # Para gráficos

# --- Configuração da Página (DEVE SER O PRIMEIRO COMANDO STREAMLIT) ---
# O tema será aplicado automaticamente pelo arquivo .streamlit/config.toml
st.set_page_config(
    layout="wide",
    page_title="Análise de Sentimento e Temática - IH", # Título que aparece na aba do navegador
    page_icon="📊"
)

# --- Inicialização do Estado da Sessão ---
if 'api_key_configured' not in st.session_state:
    st.session_state.api_key_configured = False
if 'api_key_input_value' not in st.session_state:
     st.session_state.api_key_input_value = ""

# --- Configuração da API Key ---
api_key_source = None
try:
    if hasattr(st, 'secrets') and "GOOGLE_API_KEY" in st.secrets:
        st.session_state.api_key_input_value = st.secrets["GOOGLE_API_KEY"]
        api_key_source = "secrets"
except Exception as e:
    st.sidebar.warning(f"Não foi possível ler os secrets: {e}")

# --- Interface da Barra Lateral para API Key ---
st.sidebar.header("Configuração")
if api_key_source != "secrets":
    user_provided_key = st.sidebar.text_input(
        "Insira sua Google API Key aqui:", type="password",
        key="api_key_widget", value=st.session_state.api_key_input_value
    )
    if user_provided_key != st.session_state.api_key_input_value:
         st.session_state.api_key_input_value = user_provided_key
         st.session_state.api_key_configured = False # Requer reconfiguração
else:
    st.sidebar.success("API Key carregada dos segredos!", icon="✅")
    if not st.session_state.api_key_configured: # Configura se veio dos secrets mas ainda não está marcado
         st.session_state.api_key_configured = False # Força a tentativa de configuração abaixo

# --- Tentativa de Configurar a API e o Modelo ---
model = None
if st.session_state.api_key_input_value and not st.session_state.api_key_configured:
    try:
        genai.configure(api_key=st.session_state.api_key_input_value)
        model = genai.GenerativeModel('gemini-1.5-flash')
        st.session_state.api_key_configured = True
        if api_key_source != "secrets":
             st.sidebar.success("API Key configurada!", icon="🔑")
        st.sidebar.caption(f"Modelo Gemini: gemini-1.5-flash")
    except Exception as e:
        st.sidebar.error(f"Erro API Key/Modelo. Verifique.")
        st.session_state.api_key_configured = False
        model = None
elif st.session_state.api_key_configured:
     try:
          model = genai.GenerativeModel('gemini-1.5-flash')
          #st.sidebar.caption(f"Modelo Gemini: gemini-1.5-flash (OK)") # Pode ficar repetitivo
     except Exception as e:
          st.sidebar.error(f"Erro Modelo: {e}")
          st.session_state.api_key_configured = False
          model = None

# --- Prompt Completo (Aguardando Dados para Refinamento) ---
# !! IMPORTANTE: Este prompt será atualizado quando você fornecer os dados !!
seu_prompt_completo = """
Persona: Você é uma IA Analista de Feedback de Clientes e Social Listening de uma instituição financeira (banco) Brasileira, especializada em interpretar e classificar mensagens em Português do Brasil com alta precisão. Sua tarefa exige ir além da análise literal, inferindo o contexto provável das interações (posts sobre o banco, serviços, produtos, eventos, movimentos e campanhas de marca) para classificar o sentimento (Positivo, Negativo, Neutro) e o tema da maioria das mensagens, recorrendo a Não Classificado apenas como último recurso absoluto.
Contexto: As mensagens são de clientes e público geral interagindo com posts e conteúdos de um banco (Itaú, Itaú Personnalité, Itaú Empresas, Uniclass, Itaú BBA, íon, Private) e suas iniciativas. Presuma que a maioria das mensagens curtas, emojis e interações são reações diretas ao conteúdo da marca.
Tarefa Principal: Classificar cada mensagem recebida em UMA categoria de sentimento (Positivo, Negativo, Neutro ou Não Classificado) e UMA categoria temática. Se o Sentimento for classificado como "Não Classificado", o Tema DEVE ser obrigatoriamente "Não Classificado (Tema)". Minimize a categoria "Não Classificado" interpretando ativamente o sentimento e a relevância temática sempre que possível para as demais categorias.

Definições das Categorias de Sentimento (Classificação Obrigatória - Escolha UMA)
Regra de Ouro: Antes de classificar como "Não Classificado", avalie se a mensagem, no contexto provável de uma interação com a marca/evento, pode razoavelmente ser interpreted como Positiva, Negativa ou Neutra.
Positivo
Definição: Expressões que denotam satisfação, entusiasmo, apoio, admiração, apreciação, gratidão, concordância ou engajamento positivo. Inclui reações curtas e emojis que, no contexto provável, expressam aprovação ou alegria. Inclui @menções isoladas.
Indicadores Chave: Texto de elogio ("Ótimo", "Amei", "Top!"), agradecimento ("Obrigado"), apoio ("Torcendo!"), apreciação ("Bons insights!"); Emojis positivos (😍, ❤️, 👍, 🎉, ✨, 👏, 🙏-gratidão, etc.); Menções isoladas (@username).
Negativo
Definição: Expressões que denotam insatisfação, frustração, raiva, crítica, desaprovação, reclamação, tristeza, ou qualquer relato de problemas, falhas, erros, golpes, fraudes ou experiências ruins, mesmo que factuais ou com pedido vago de ajuda.
Indicadores Chave: Texto de crítica ("Péssimo", "Lixo"), relato de problema ("Não funciona", "App travado", "Cobrança irregular", "Fui vítima de golpe"), reclamação ("Atendimento ruim"), insatisfação ("Taxa abusiva"), frustração (CAIXA ALTA, !!!), advertência ("Não recomendo"); Emojis negativos (😠, 😡, 👎, 😢, etc.). Qualquer menção textual a um problema ou evento grave é Negativa.
Neutro
Definição: Mensagens que buscam/fornecem informação factual, fazem observações objetivas, ou expressam reações sem forte valência positiva ou negativa, assumindo relevância contextual e não relatando problemas.
Indicadores Chave: Texto de pergunta/solicitação ("Como faço?", "Qual o endereço?"), declaração factual ("O evento é em Miami"), sugestão objetiva, resposta curta factual ("Ok"); Emojis neutros (🤔, 👀, 😂, 😅, 🙏-"por favor", etc.).
Último Recurso:
Não Classificado
Definição: Aplicar SOMENTE quando a mensagem for impossível de classificar como Positiva, Negativa ou Neutra devido a UM destes motivos:
1.  Idioma Estrangeiro: Predominantemente não em Português. (Ex: "What time is it?") -> Resulta em Sentimento: Não Classificado, Tema: Não Classificado (Tema).
2.  Incompreensível: Erros graves, códigos aleatórios, texto sem sentido. (Ex: "asdfghjkl") -> Resulta em Sentimento: Não Classificado, Tema: Não Classificado (Tema).
3.  Spam Óbvio: Conteúdo repetitivo claro, links suspeitos isolados, promoções não relacionadas. -> Resulta em Sentimento: Não Classificado, Tema: Não Classificado (Tema).
4.  Totalmente Off-Topic e Sem Conexão: Assuntos completamente alheios ao universo da marca/evento. -> Resulta em Sentimento: Não Classificado, Tema: Não Classificado (Tema).
5.  Interações Sociais Textuais Puras e Genéricas ISOLADAS: Saudações isoladas ("Bom dia", "Olá"), risadas textuais isoladas ("kkkk", "rsrs") sem QUALQUER outro elemento interpretável. (Ex: apenas "kkkkkk") -> Resulta em Sentimento: Não Classificado, Tema: Não Classificado (Tema). MUITO IMPORTANTE: Se houver "kkkk" junto com outra frase (ex: "kkkk adorei"), classifique a frase principal.
Regra Vinculada: Se o sentimento for classificado aqui, o tema será obrigatoriamente "Não Classificado (Tema)".

Definições das Categorias Temáticas (Classificação Obrigatória - Escolha UMA)
***IMPORTANTE: Você DEVE usar EXATAMENTE UM dos nomes de Tema listados abaixo numerados de 1 a 9. Não invente nomes novos ou variações.***
Regra: Se o Sentimento for "Não Classificado", o Tema é "Não Classificado (Tema)". Para os demais sentimentos (P/N/N), atribua o tema mais específico possível da lista abaixo.

1.  **Elogio Geral (Marca/Evento/Conteúdo/Experiência)**: Elogios textuais ou via emoji positivo sobre a marca, evento, post, experiência geral. (Sentimento: Positivo)
2.  **Elogio Específico (Pessoa/Figura Pública/Representante/"Laranjinha")**: Elogios a indivíduos, atletas, iniciativas nomeadas. (Sentimento: Positivo)
3.  **Reclamação/Crítica (Serviços/Produtos/Atendimento/Políticas)**: Reclamações, críticas, relatos de problemas (incluindo golpes/fraudes) sobre aspectos do banco. (Sentimento: Negativo)
4.  **Problemas Técnicos (Plataformas/Funcionalidades)**: Relatos de problemas com app, site, maquininha, etc. (Sentimento: Negativo)
5.  **Apoio/Incentivo (Pessoas/Causas/Marca)**: Mensagens de torcida, apoio, incentivo. Pode incluir emojis positivos contextuais. (Sentimento: Positivo)
6.  **Solicitação/Dúvida/Sugestão**: Perguntas, pedidos de informação, sugestões. (Sentimento: Neutro)
7.  **Interação Social**: Aplicar apenas quando o sentimento for P, N ou Neutro. Usar para: Emojis isolados P/N/N sem contexto específico forte, @menções isoladas (Positivo). Se um emoji/menção P/N/N pode ter tema mais específico pelo contexto (ex: 🏆 em post de vitória -> Apoio), priorize o tema específico. Não usar se sentimento for Não Classificado.
8.  **Discussão Específica (Tópico da Campanha/Evento)**: Comentários sobre o tema central (jogo, jogador, detalhe do evento), incluindo observações factuais. (Sentimento: Pode ser Positivo, Negativo ou Neutro)
9.  **Não Classificado (Tema)**: Aplicado exclusivamente e obrigatoriamente quando o sentimento também for "Não Classificado". Engloba mensagens nos critérios 1 a 5 da seção "Não Classificado" de Sentimento. (Sentimento: Não Classificado)

Instruções Finais de Classificação:
1.  Análise Dupla Obrigatória: Sentimento + Tema.
2.  Idioma Primeiro: Se não for predominantemente Português, a resposta DEVE SER: Sentimento: Não Classificado, Tema: Não Classificado (Tema).
3.  "kkkkk" Isolado: Se a mensagem for APENAS "kkkk", "rsrs" ou similar, a resposta DEVE SER: Sentimento: Não Classificado, Tema: Não Classificado (Tema).
4.  Priorize P/N/N: Esforce-se para encontrar um sentimento Positivo, Negativo ou Neutro antes de usar Não Classificado.
5.  Verificação Final de Sentimento: Se, após avaliar P/N/N, a mensagem se encaixar nos critérios 2, 3 ou 4 de "Não Classificado" de Sentimento (Incompreensível, Spam, Off-topic), atribua Sentimento = Não Classificado.
6.  Vinculação de Tema NC: Se o Sentimento for "Não Classificado", o Tema é AUTOMATICAMENTE "Não Classificado (Tema)".
7.  Atribuição de Tema (para P/N/N): Se o sentimento for P, N ou N, escolha o tema mais específico possível USANDO EXATAMENTE UM dos nomes da lista numerada de 1 a 9 acima.
8.  Mensagens Mistas: Classifique pelo elemento predominante (Reclamação > outros; Pergunta > outros).

Formato de Resposta OBRIGATÓRIO:
Responda APENAS com as duas linhas abaixo, usando EXATAMENTE os nomes de categorias definidos:
Sentimento: [Nome Exato da Categoria de Sentimento]
Tema: [Nome Exato da Categoria de Tema]

***NÃO inclua nenhuma outra palavra, explicação ou formatação na sua resposta.***

Agora, analise a seguinte mensagem:
{comment}
"""

# --- Listas de Categorias Válidas ---
categorias_sentimento_validas = ["Positivo", "Negativo", "Neutro", "Não Classificado"]
categorias_tema_validas = [
    "Elogio Geral (Marca/Evento/Conteúdo/Experiência)",
    "Elogio Específico (Pessoa/Figura Pública/Representante/\"Laranjinha\")",
    "Reclamação/Crítica (Serviços/Produtos/Atendimento/Políticas)",
    "Problemas Técnicos (Plataformas/Funcionalidades)",
    "Apoio/Incentivo (Pessoas/Causas/Marca)",
    "Solicitação/Dúvida/Sugestão",
    "Interação Social",
    "Discussão Específica (Tópico da Campanha/Evento)",
    "Não Classificado (Tema)"
]
categorias_erro = ["Erro Parsing", "Erro API"]
categorias_erro_tema_especifico = ["Erro API (Timeout)", "Erro API (Geral)", "Erro API (Modelo não iniciado)"]
# Combina todas as categorias de erro para facilitar a filtragem
todas_categorias_erro = list(set(categorias_erro + categorias_erro_tema_especifico))
# Categorias a serem EXCLUÍDAS dos gráficos
categorias_excluir_sentimento = ["Não Classificado"] + todas_categorias_erro
categorias_excluir_tema = ["Não Classificado (Tema)"] + todas_categorias_erro

# --- Função para Analisar um Comentário ---
def analisar_comentario(comentario, modelo_gemini):
    """Envia um comentário para a API Gemini e retorna o sentimento e tema classificados."""
    if not comentario or not isinstance(comentario, str) or comentario.strip() == "":
         return "Não Classificado", "Não Classificado (Tema)"
    if not modelo_gemini:
        # st.warning(f"Tentativa de análise sem modelo Gemini inicializado para: '{comentario[:50]}...'") # Pode poluir muito
        return "Erro API", "Erro API (Modelo não iniciado)"

    prompt_com_comentario = seu_prompt_completo.format(comment=comentario)
    try:
        safety_settings = { "HARM_CATEGORY_HARASSMENT": "BLOCK_NONE", "HARM_CATEGORY_HATE_SPEECH": "BLOCK_NONE",
                           "HARM_CATEGORY_SEXUALLY_EXPLICIT": "BLOCK_NONE", "HARM_CATEGORY_DANGEROUS_CONTENT": "BLOCK_NONE"}
        request_options = {"timeout": 60}
        response = modelo_gemini.generate_content(prompt_com_comentario, safety_settings=safety_settings, request_options=request_options)

        texto_resposta = response.text.strip()
        sentimento_extraido = "Erro Parsing"; tema_extraido = "Erro Parsing"
        linhas = texto_resposta.split('\n')
        for linha in linhas:
            linha_strip = linha.strip()
            if linha_strip.lower().startswith("sentimento:"): sentimento_extraido = linha_strip.split(":", 1)[1].strip()
            elif linha_strip.lower().startswith("tema:"): tema_extraido = linha_strip.split(":", 1)[1].strip()

        if sentimento_extraido == "Erro Parsing" or tema_extraido == "Erro Parsing":
             st.warning(f"Formato resp. inesperado '{comentario[:50]}...': '{texto_resposta}'")
             return "Erro Parsing", "Erro Parsing"

        if sentimento_extraido not in categorias_sentimento_validas:
            st.warning(f"Sent. inválido '{sentimento_extraido}': '{comentario[:50]}...'")
            return "Erro Parsing", "Erro Parsing"

        if sentimento_extraido == "Não Classificado":
            if tema_extraido != "Não Classificado (Tema)":
                #st.warning(f"Correção: Sent='NC' mas Tema='{tema_extraido}'. Ajustado. Msg:'{comentario[:50]}...'")
                return "Não Classificado", "Não Classificado (Tema)"
            else: return "Não Classificado", "Não Classificado (Tema)"
        else: # Sentimento é P, N ou Neutro
             if tema_extraido not in categorias_tema_validas or tema_extraido == "Não Classificado (Tema)":
                  st.warning(f"Tema inválido '{tema_extraido}' p/ Sent='{sentimento_extraido}'. Msg:'{comentario[:50]}...'")
                  return sentimento_extraido, "Erro Parsing"
             else: return sentimento_extraido, tema_extraido

    except Exception as e:
        if "timeout" in str(e).lower(): st.error(f"Timeout API: '{comentario[:50]}...'"); return "Erro API", "Erro API (Timeout)"
        else: st.error(f"Erro API Geral: '{comentario[:50]}...'."); return "Erro API", "Erro API (Geral)"

# --- Interface Principal ---
st.title("📊 Aplicativo para análise de sentimento e temática automatizado por IA")
st.markdown("Este aplicativo foi desenvolvido pelo time de Social Intelligence do Hub de Inovação da Ihouse para o Itaú. As análises são realizadas e geradas através do Gemini.")
st.markdown("---") # Linha divisória

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
            df.dropna(subset=['conteúdo'], inplace=True)
            df = df[df['conteúdo'].astype(str).str.strip() != '']
            total_comentarios_validos = len(df)
    except Exception as e: st.error(f"Erro ao ler o arquivo: {e}"); df = None

if df is not None:
    st.subheader("Pré-visualização dos dados:")
    st.dataframe(df.head(10))
    st.info(f"Total de comentários válidos (não vazios) encontrados: **{total_comentarios_validos}**")

    # Container para mostrar após a análise
    results_container = st.container()

    # Armazenar resultados no estado da sessão para persistir
    if 'analysis_done' not in st.session_state: st.session_state.analysis_done = False
    if 'df_results' not in st.session_state: st.session_state.df_results = None

    if analisar_btn:
        if total_comentarios_validos == 0: st.warning("Nenhum comentário válido para análise.")
        elif not model: st.error("Erro: Modelo Gemini não inicializado. Verifique a API Key.")
        else:
            with st.spinner(f"Analisando {total_comentarios_validos} comentários... Isso pode levar alguns minutos."):
                progress_bar = st.progress(0)
                status_text = st.empty()
                resultados_sentimento = []; resultados_tema = []
                df_copy = df.copy() # Trabalhar com cópia para não afetar o df original da visualização

                for i, comentario in enumerate(df_copy['conteúdo']):
                    sentimento, tema = analisar_comentario(str(comentario), model)
                    resultados_sentimento.append(sentimento)
                    resultados_tema.append(tema)
                    progresso = (i + 1) / total_comentarios_validos
                    progress_bar.progress(progresso)
                    status_text.text(f"Analisando: {i+1}/{total_comentarios_validos}")
                    #time.sleep(0.01) # Pausa mínima opcional

                progress_bar.empty(); status_text.success(f"✅ Análise concluída!")
                df_copy['Sentimento_Classificado'] = resultados_sentimento
                df_copy['Tema_Classificado'] = resultados_tema
                st.session_state.df_results = df_copy # Salva resultados no estado
                st.session_state.analysis_done = True
                st.rerun() # Força o rerodamento para exibir os resultados abaixo

    # Exibe os resultados se a análise foi concluída
    if st.session_state.analysis_done and st.session_state.df_results is not None:
        df_results = st.session_state.df_results
        total_analisados = len(df_results)

        with results_container:
            st.markdown("---")
            st.subheader("Visualização dos Resultados")

            # --- Gráficos (em colunas) ---
            col1, col2 = st.columns(2)

            with col1:
                st.markdown("##### Distribuição de Sentimento")
                # Filtrar para gráfico de sentimento
                df_sent_chart = df_results[~df_results['Sentimento_Classificado'].isin(categorias_excluir_sentimento)]
                sent_counts_chart = df_sent_chart['Sentimento_Classificado'].value_counts()
                total_sent_chart = sent_counts_chart.sum()

                if total_sent_chart > 0:
                    sent_perc_chart = (sent_counts_chart / total_sent_chart * 100)
                    df_plot_sent = pd.DataFrame({'Sentimento': sent_counts_chart.index, 'Volume': sent_counts_chart.values, 'Percentual': sent_perc_chart.values})

                    fig_sent = px.pie(df_plot_sent, names='Sentimento', values='Volume', hole=0.4,
                                      color='Sentimento',
                                      color_discrete_map={'Positivo': 'green', 'Negativo': 'red', 'Neutro': 'gold'}, # Amarelo = gold
                                      title='Sentimentos (Excluindo Não Classificado)')
                    # Ajustar texto do hover e labels
                    fig_sent.update_traces(textposition='inside',
                                           textinfo='percent+label',
                                           hovertemplate="<b>%{label}</b><br>Volume: %{value}<br>Percentual: %{percent:.1%}<extra></extra>")
                    fig_sent.update_layout(showlegend=False, title_x=0.5, height=350, margin=dict(l=10, r=10, t=40, b=10))
                    st.plotly_chart(fig_sent, use_container_width=True)
                else:
                    st.warning("Nenhum sentimento válido (Positivo/Negativo/Neutro) classificado para exibir o gráfico.")

            with col2:
                st.markdown("##### Distribuição Temática")
                 # Filtrar para gráfico de tema
                df_tema_chart = df_results[~df_results['Tema_Classificado'].isin(categorias_excluir_tema)]
                tema_counts_chart = df_tema_chart['Tema_Classificado'].value_counts()
                total_tema_chart = tema_counts_chart.sum()

                if total_tema_chart > 0:
                    tema_perc_chart = (tema_counts_chart / total_tema_chart * 100)
                    df_plot_tema = pd.DataFrame({'Tema': tema_counts_chart.index, 'Volume': tema_counts_chart.values, 'Percentual': tema_perc_chart.values})
                    # Ordenar por Volume para o gráfico
                    df_plot_tema = df_plot_tema.sort_values(by='Volume', ascending=False)

                    fig_tema = px.bar(df_plot_tema, x='Tema', y='Volume',
                                      color_discrete_sequence =['#FFA500']*len(df_plot_tema), # Laranja para todas as barras
                                      title='Temas (Excluindo Não Classificado)',
                                      text=df_plot_tema.apply(lambda row: f"{row['Volume']}<br>({row['Percentual']:.1f}%)", axis=1)) # Texto com volume e %
                    fig_tema.update_traces(textposition='outside') # Coloca o texto fora da barra
                    fig_tema.update_layout(xaxis_title=None, yaxis_title="Volume Bruto", title_x=0.5, height=350, margin=dict(l=10, r=10, t=40, b=10))
                    # Rotacionar labels do eixo X se forem muitos temas
                    fig_tema.update_xaxes(tickangle= -45)
                    st.plotly_chart(fig_tema, use_container_width=True)
                else:
                     st.warning("Nenhum tema válido classificado para exibir o gráfico.")


            # --- Tabelas Agregadas (Como antes) ---
            st.markdown("---")
            st.subheader("Tabelas de Resumo")
            col_t1, col_t2 = st.columns(2)

            with col_t1:
                st.markdown("###### Tabela 1: Análise de Sentimento (Completa)")
                todas_cats_sent = categorias_sentimento_validas + todas_categorias_erro
                sent_counts = df_results['Sentimento_Classificado'].value_counts().reindex(todas_cats_sent, fill_value=0)
                sent_perc = (sent_counts / total_analisados * 100).round(2) if total_analisados > 0 else 0
                tabela_sent = pd.DataFrame({'Sentimento': sent_counts.index, 'Volume Bruto': sent_counts.values, 'Percentual (%)': sent_perc.values})
                total_sent = pd.DataFrame({'Sentimento': ['Total'], 'Volume Bruto': [total_analisados], 'Percentual (%)': [100.0]})
                tabela_sent = pd.concat([tabela_sent[tabela_sent['Volume Bruto'] > 0], total_sent], ignore_index=True) # Mostra só linhas com valor > 0 + Total
                st.table(tabela_sent.style.format({'Percentual (%)': '{:.2f}%'}))

            with col_t2:
                st.markdown("###### Tabela 2: Análise Temática (Completa)")
                todas_cats_tema = categorias_tema_validas + todas_categorias_erro
                tema_counts = df_results['Tema_Classificado'].value_counts().reindex(todas_cats_tema, fill_value=0)
                tema_counts = tema_counts[~tema_counts.index.duplicated(keep='first')] # Remove duplicatas caso existam
                tema_perc = (tema_counts / total_analisados * 100).round(2) if total_analisados > 0 else 0
                tabela_tema = pd.DataFrame({'Tema': tema_counts.index, 'Volume Bruto': tema_counts.values, 'Percentual (%)': tema_perc.values})
                total_tema = pd.DataFrame({'Tema': ['Total'], 'Volume Bruto': [total_analisados], 'Percentual (%)': [100.0]})
                tabela_tema = pd.concat([tabela_tema[tabela_tema['Volume Bruto'] > 0], total_tema], ignore_index=True) # Mostra só linhas com valor > 0 + Total
                st.table(tabela_tema.style.format({'Percentual (%)': '{:.2f}%'}))

            # --- Tabela Completa e Download ---
            st.markdown("---")
            st.subheader("Resultados Completos Detalhados")
            st.dataframe(df_results)

            @st.cache_data
            def convert_df_to_csv(df_conv): return df_conv.to_csv(index=False).encode('utf-8-sig')
            csv_output = convert_df_to_csv(df_results)
            st.download_button("💾 Download Resultados (.csv)", csv_output, 'analise_gemini_resultados.csv', 'text/csv', key='download_csv')

elif not uploaded_file and not analisar_btn :
     st.info("⬅️ Faça o upload de um arquivo .csv ou .xlsx na barra lateral para começar.")