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
if 'insights_generated' not in st.session_state: st.session_state.insights_generated = None

# --- Configuração da API Key ---
api_key_source = None
try:
    if hasattr(st, 'secrets') and "GOOGLE_API_KEY" in st.secrets:
        st.session_state.api_key_input_value = st.secrets["GOOGLE_API_KEY"]
        api_key_source = "secrets"
except Exception as e:
    st.sidebar.warning(f"Não foi possível ler os secrets: {e}", icon="⚠️")

# --- Interface da Barra Lateral para API Key ---
st.sidebar.header("Configuração")
if api_key_source != "secrets":
    user_provided_key = st.sidebar.text_input(
        "Insira sua Google API Key aqui:", type="password",
        key="api_key_widget", value=st.session_state.api_key_input_value,
        help="Sua chave da API do Google AI Studio (Gemini)."
    )
    if user_provided_key != st.session_state.api_key_input_value:
         st.session_state.api_key_input_value = user_provided_key
         st.session_state.api_key_configured = False
         st.session_state.analysis_done = False
         st.session_state.insights_generated = None
         st.rerun()
else:
    st.sidebar.success("API Key carregada dos segredos!", icon="✅")
    if not st.session_state.api_key_configured:
        st.session_state.api_key_configured = False

# --- Tentativa de Configurar a API e o Modelo ---
model = None
if st.session_state.api_key_input_value and not st.session_state.api_key_configured:
    try:
        genai.configure(api_key=st.session_state.api_key_input_value)
        model = genai.GenerativeModel('gemini-1.5-flash')
        st.session_state.api_key_configured = True
        if api_key_source != "secrets":
            st.sidebar.success("API Key configurada com sucesso!", icon="🔑")
        st.sidebar.caption(f"Modelo Gemini: gemini-1.5-flash")
    except Exception as e:
        st.sidebar.error(f"Erro ao configurar API Key/Modelo. Verifique a chave.", icon="🚨")
        st.session_state.api_key_configured = False
        model = None
elif st.session_state.api_key_configured:
     try:
         model = genai.GenerativeModel('gemini-1.5-flash')
     except Exception as e:
         st.sidebar.error(f"Erro ao recarregar o Modelo: {e}", icon="🚨")
         st.session_state.api_key_configured = False
         model = None


# --- Prompt Principal REFINADO v6 (Baseado na Análise Comparativa e Feedback v2) ---
seu_prompt_completo = """
Persona: Você é uma IA Analista de Feedback de Clientes e Social Listening altamente especializada no setor bancário brasileiro, com profundo conhecimento sobre o Banco Itaú e seu ecossistema de marcas (Itaú, Personnalité, Uniclass, Empresas, íon, Private, BBA, Itubers). Você compreende produtos (CDB, LCI, Pix), jargões (TED, DOC) e o contexto de campanhas (influenciadores, eventos como Rock in Rio, The Town). Sua análise combina rigor na aplicação das regras com compreensão contextual.

Objetivo:
1.	Primário: Classificar CADA mensagem recebida em Português do Brasil (Pt-BR) em UMA categoria de Sentimento e UMA categoria Temática, aderindo ESTRITAMENTE às definições, regras e prioridades abaixo.
2.	Secundário: (Este prompt foca APENAS na classificação).

Contexto Geral: Mensagens de clientes/público sobre posts/conteúdos do Itaú e submarcas (produtos, serviços, atendimento, plataformas, campanhas, patrocínios, imagem). Reações curtas são contextuais ao post original.

=== REGRAS GERAIS E DE OURO ===
1.  Análise Dupla Obrigatória: Sentimento + Tema para cada mensagem.
2.  **APLIQUE AS REGRAS DE "NÃO CLASSIFICADO" PRIMEIRO:** Se a mensagem se encaixar em QUALQUER critério de Não Classificado (ver seção NC abaixo), classifique IMEDIATAMENTE como Sentimento: Não Classificado, Tema: Não Classificado (Tema) e NÃO prossiga. Só avance para P/N/Neutro se NENHUM critério NC se aplicar.
3.  Priorize P/N/Neutro: Se não for NC, use as definições abaixo.
4.  Vinculação NC Estrita: Se Sentimento = Não Classificado, Tema DEVE SER SEMPRE Não Classificado (Tema). SEM EXCEÇÕES.
5.  Foco no Conteúdo Relevante: Ignore ruídos como saudações isoladas no início/fim se houver conteúdo principal. Classifique com base na intenção principal da mensagem.

=== NÃO CLASSIFICADO (APLICAR PRIMEIRO!) ===
4.	Não Classificado: APLICAR **OBRIGATORIAMENTE E PRIORITARIAMENTE** SE A MENSAGEM ATENDER A UM DESTES CRITÉRIOS (ANTES de tentar P/N/Neutro):
    1.	**Idioma Estrangeiro (predominante):** Ex: "What time?", "gracias por venir", "Do not sleep on solta o pavo".
    2.	**Incompreensível:** Erros graves, digitação aleatória, sem sentido lógico. Ex: "asdf ghjk", "L0p9l9", "Kbut", ".oitoitoitameiamei", emojis sem sentido óbvio juntos se isolados, `^`, `>>`.
    3.	**Menção @ Isolada ou Menção + Texto Incompleto/Ambíguo:** Contém APENAS o símbolo `@` isolado, ou menção (@ ou []) mas o texto acompanhante é incompreensível, ambíguo demais para classificar, ou apenas uma letra/número/emoji sem contexto claro. Ex: `"@"` (como única mensagem), `@itau p`, `@itau 2`, `@itau ...`, `@2🥺`. **NÃO inclui menção a usuário isolada (ex: `@pedrotavares`), que é Positivo/Interação.**
    4.	**Spam/Link Isolado:** Conteúdo repetitivo óbvio, promoções não relacionadas, propaganda de terceiros, URL isolada SEM contexto relevante ou explicação. Ex: "Confira: https://...", "https://t.co/xxxxx", "Buy #Bitcoin 👍", "Apostas Grátis...", "@c................:não traduza...".
    5.	**Totalmente Off-Topic:** Assunto sem QUALQUER conexão clara com Itaú, bancos, finanças, produtos/serviços financeiros, a campanha/evento em questão, ou figuras públicas associadas. Ex: "Receita de bolo", "Anistia já!", "É movimento pro Alckmin assumir?", "@que Deus abençoe vocês...", "ESTOU COM DOR ORE POR MIM", "@moço de onde é esse calendário?", "Trump está chegando...", "Vocês atendem a N.O.M.", comentário sobre time de futebol não relacionado a patrocínio, "@Não, obrigado... De repetição já ch ga minha mulher reclamando.", "Hermes trismegisto...", "Fernanda Torres é Truong My Lan.".
    6.	**Interação Social Pura Textual/Emoji Isolada:** Mensagem contém APENAS saudações/despedidas ("Bom dia", "Boa noite amigo", "@BOA TADE", "Oi", "Tchau"), APENAS risadas textuais ("kkkk", "rsrs"), APENAS agradecimentos/expressões religiosas genéricas isoladas ("Obg", "Amem", "@amem"), APENAS concordâncias curtas isoladas ("Isso aí"), ou APENAS emojis de interação social sem outro conteúdo ou símbolos de pontuação isolados. Ex: "@amem", "@boa noite", "kkkk", "👍" (isolado), "❤️" (isolado), `☺️` (isolado), `@Oi?`, `!!!!!!!!!!!!!!` (isolado). *NÃO aplicar se a interação acompanha conteúdo classificável (Ex: "kkkk adorei" -> classificar "adorei").*

=== DEFINIÇÕES DE SENTIMENTO (Escolha UMA, APÓS verificar regras NC) ===

1.	Positivo: Expressa satisfação, apoio, entusiasmo, gratidão genuína, apreciação (mesmo moderada "interessante"), concordância clara, ou engajamento positivo explícito, **incluindo menção a usuário isolada.**
    *   Indicadores: **Menção a usuário isolada (Ex: `@pedrotavares`, `[Luiz Erik]`) - SEMPRE Positivo/Interação Social.** Elogios claros ("Amei", "Top", "Excelente", "Maravilhoso", "Grande Mestre", "Melhor propaganda"), Agradecimentos específicos ("Obrigado por trazerem o show"), Apoio/Torcida ("Parabéns pela iniciativa", "QUEREMOS TURNÊ!", "Vocês são os MAIORAIS"), Apreciação ("Belo post", "Interessante", "De arrepiar"), Concordância explícita com algo positivo ("Concordo, ele é gênio!"); Emojis claramente positivos isolados ou acompanhando texto positivo (😍, ❤️, 👍, 🎉, ✨, 👏); Combinações Texto/Emoji Positivo. **Focar no ponto principal em mensagens mistas (Ex: "@itau Alguem q investe em musica de verdade e nao essa bosta de sertanejo... Salve Jorge!!!" -> Foco no "investe em musica de verdade/Salve Jorge" -> Positivo).**

2.	Negativo: Expressa insatisfação, crítica, raiva, frustração, reclamação, tristeza, **sarcasmo óbvio**, ou **afirmação/relato direto** de problema, falha, erro, golpe, fraude ou experiência ruim.
    *   Indicadores: Críticas diretas ("Péssimo", "Banco lixo", "Que bosta"), **relato/afirmação de problemas ("Não funciona CDB", "Cobrança indevida DOC", "Fui vítima de golpe Pix", "@itau Erro TED", "não estava conseguindo acessar o app", "@itau Cuidado com o app do @itau! Banco não computa...")**, reclamações diretas ("Atendimento Uniclass ruim", "Atendimento horrível"), insatisfação direta ("Taxa alta BBA", "Quero taxa mais baixa na minha conta.", "Pago um monte de taxas..."), frustração (CAIXA ALTA negativa), advertência ("Não recomendo íon"); **Sarcasmo óbvio (Ex: "@itau São 86 anos de Alquimia!...Pedra Filosofal!", "Que ótimo, o app caiu de novo")**; **Provocação/Comparação com concorrente (Ex: "@abibfilho na dúvida vou chamar o @bancodobrasil...")**; Emojis claramente negativos (😠, 😡, 👎, 😢, 💩, 🤮). Qualquer menção a golpe, fraude, erro funcional grave é Negativa. Comentários sobre política/governo associados negativamente ao banco ("Banco lixo apoiador de corruptos", "Banco da lacração", "Banco maldito", "Banco comunista").

3.	Neutro: Busca/fornece informação, observação factual, **pergunta** (mesmo sobre problemas), **sugestão**, **pedido**, expressão de equilíbrio, ou reação ambígua/sem forte valência P/N por padrão. **Pedidos/Sugestões/Perguntas são GERALMENTE Neutros, mesmo com emojis positivos.**
    *   Indicadores: **Perguntas (incluindo sobre problemas: "Como faço LCI?", "Quando terá mais show?", "O que isso tem a ver com banco?", "@itau @jorgebenjor divo o app de vcs ta fora do ar?", "@itau oloko ele ainda tá vivo?", "@itau Os ruanistas?", "@cabedelos show em cabedelo*")**; Respostas a perguntas; **Pedidos/Sugestões diretas ("@itau Aumenta meu limite 👍", "@itau tragam #technotronic", "Me da dinheiro Itaú kkkk", "Me da um emprego🙏", "@itau Itaú ? Faz uma publi com o Davi", "@itau ITAÚ ME LEVA PRO THE TOWN", "@tatinhagrassi pede pra ele um show no rio também")**; Expressões de equilíbrio ("Ok"), sugestões implícitas ("Poderia ser melhor o app Personnalité"); Pedidos de ajuda claros (sem problema grave); Informações factuais ("O Rock in Rio é patrocinado"); Respostas curtas factuais ("Entendido"); Emojis ambíguos padrão isolados (🙏, 😂, 🤔, 👀, `[👈😀👈]`); Termos/siglas específicos ("ESG", "Les alchimistes"); Avisos/Declarações factuais ("@itau JORGE BEN JOR NAO DEIXE A POLITICA TE USAR", "Não conheço esse país").

=== DEFINIÇÕES DE TEMA (Escolha UMA - Aplicar Regras de Prioridade Abaixo, SOMENTE SE NÃO FOR NC) ===
***IMPORTANTE: Use EXATAMENTE um dos nomes de Tema 1 a 9 abaixo. Se Sentimento = Não Classificado, Tema = Não Classificado (Tema).***

1.	Marca e Imagem: Percepção geral da marca Itaú ou submarcas, reputação, campanhas institucionais, patrocínios gerais. (Sentimento: P/N/Neutro)
2.	Produtos e Serviços (Geral): Sobre cartões, contas, seguros, investimentos (CDB, LCI, íon), crédito, taxas, políticas, benefícios, contratação, cancelamento. (Sentimento: P/N/Neutro)
3.	Atendimento e Suporte: Sobre canais (agência, telefone, chat), qualidade do suporte, resolução de problemas pelo atendimento. (Sentimento: P/N/Neutro)
4.	Plataformas Digitais (App/Site/ATM): Feedback sobre usabilidade, design, funcionalidades (PIX, TED, DOC, login), performance de app, site, caixas eletrônicos. (Sentimento: P/N/Neutro)
5.	Figuras Públicas e Representantes: Foco em atletas, influenciadores, creators, "laranjinhas", executivos, artistas (Jorge Ben Jor na campanha) associados a campanhas ou à marca. (Sentimento: P/N/Neutro)
6.	Eventos e Campanhas Específicas: Discussões focadas em evento/campanha nomeado (logística, experiência, tema). (Sentimento: P/N/Neutro)
7.	Segurança e Fraude: Sobre golpes, fraudes, segurança da conta, phishing, roubos, cobranças indevidas graves. (Sentimento: Geralmente Negativo, pode ser Neutro)
8.	**Solicitação/Dúvida/Sugestão (Transversal):** Prioridade média. Usar quando o FOCO PRINCIPAL da mensagem (Sentimento Neutro) é uma pergunta, pedido ou sugestão sobre QUALQUER tema (produto, serviço, evento, plataforma, etc.). Ex: "App fora do ar?", "Aumenta meu limite", "Faz publi com Davi". (Sentimento: **Neutro**)
9.	**Interação Social e Engajamento:** Prioridade MÍNIMA. Usar SOMENTE para: **Menção a usuário isolada (@username) - SEMPRE Positivo**; Emojis P/N/Neutro ISOLADOS sem outro tema claro; Múltiplas menções (@user1 @user2) SEM texto adicional. (Sentimento: Positivo para @username isolado, P/N/Neutro para outros casos).
10.	Não Classificado (Tema): Exclusivamente quando Sentimento = Não Classificado.

=== REGRAS DE PRIORIDADE PARA TEMAS (Aplicar SOMENTE SE NÃO FOR NC) ===
Aplique na seguinte ordem. Se a mensagem se encaixar em múltiplos temas, escolha o primeiro da lista que se aplicar:
1.	Segurança e Fraude: (Prioridade Máxima) Se mencionar golpe, fraude, segurança, cobrança indevida grave.
2.	Plataformas Digitais (App/Site/ATM): Se o feedback (P/N/Neutro - *exceto se for SÓ pergunta/pedido/sugestão*) for especificamente sobre essas plataformas.
3.	Atendimento e Suporte: Se o foco (P/N/Neutro - *exceto se for SÓ pergunta/pedido/sugestão*) for a interação com canais de atendimento.
4.	Produtos e Serviços (Geral): Se sobre características, taxas, contratação/cancelamento (P/N/Neutro - *exceto se for SÓ pergunta/pedido/sugestão*).
5.	Eventos e Campanhas Específicas: Se claramente focado em um evento/campanha nomeado.
6.	Figuras Públicas e Representantes: Se o foco principal for a pessoa/representante.
7.	Marca e Imagem: Para comentários gerais sobre a marca/reputação/patrocínios gerais.
8.	**Solicitação/Dúvida/Sugestão (Transversal):** Se o foco principal for a pergunta/pedido/sugestão em si (Sentimento Neutro).
9.	**Interação Social e Engajamento:** Para @username isolado (Positivo), emojis isolados, múltiplas menções isoladas. (Prioridade Mínima).
10.	Não Classificado (Tema): Apenas se Sentimento = Não Classificado.

=== INSTRUÇÕES ADICIONAIS DE CLASSIFICAÇÃO ===
*   Formato de Resposta: EXATAMENTE DUAS LINHAS, SEMPRE:
    Sentimento: [Nome Exato da Categoria de Sentimento]
    Tema: [Nome Exato da Categoria de Tema]
    (Não inclua NADA MAIS).
*   **Priorize NÃO CLASSIFICADO:** Verifique TODAS as regras de NC primeiro. Se alguma aplicar, use NC/NC(Tema) e PARE.
*   Aplicar Prioridade de Tema: Se não for NC, siga estritamente as regras de prioridade de tema.
*   Detectar Sarcasmo/Ironia: Tentar identificar (contradições, elogios exagerados, comparações negativas com concorrentes) e classificar como **Negativo**. Tema segue prioridade normal. Ex: "@itau São 86 anos...", "@abibfilho na dúvida vou chamar o @bancodobrasil...".
*   **Menções:**
    *   `@` isolado ou `@` + texto incompreensível/ambíguo -> **Não Classificado / Não Classificado (Tema)** (Regra NC 3).
    *   `@username` isolado -> **Positivo / Interação Social e Engajamento**.
    *   Menção + Texto claro -> Classificar com base no TEXTO.
*   Emojis Ambíguos/Positivos em Pedidos: Emojis (👍, 😀) em pedidos/sugestões/perguntas NÃO tornam o sentimento Positivo. O sentimento nesses casos é **Neutro**.
*   **Perguntas/Respostas:** Geralmente **Neutro**. Usar Tema "Solicitação/Dúvida/Sugestão" se for pergunta/pedido direto, ou tema relevante ao assunto da pergunta. **Perguntas sobre problemas ("App fora do ar?") são Neutras.**
*   Emojis Mistos: Prioridade Sentimento: Negativo > Positivo > Neutro.
*   Ênfase (!!!, ???): Modifica/reforça sentimento base. Isolado (`!!!!!!!!!!!!!!`) -> **Não Classificado / Não Classificado (Tema)** (Regra NC 6).
*   **Mensagens Mistas:** Classifique pelo elemento PREDOMINANTE ou FOCO PRINCIPAL. (Reclamação/Problema/Fraude/Sarcasmo > Elogio > Pergunta/Sugestão). Ex: Critica sertanejo mas elogia Itaú/Jorge Ben -> **Positivo / Figuras Públicas ou Marca e Imagem**.

Agora, classifique a seguinte mensagem:
{comment}
"""


# --- Prompt para Geração de Insights (NOVO) ---
# ... (sem alterações aqui, mantenha como estava)
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
*   (Analise a proporção de comentários Neutros. Se o tema 'Solicitação/Dúvida/Sugestão' aparecer nos Top Temas, indique oportunidade de esclarecimento ou melhoria com base no alto volume de perguntas/pedidos. Se temas negativos recorrentes aparecerem, sugira investigação ou ação específica para mitigar.)

### Observações Gerais:
*   (Faça um balanço geral. Comente se a distribuição de sentimentos parece saudável ou preocupante. Mencione se a proporção de 'Não Classificado/Erros' é alta, indicando possíveis problemas na coleta ou classificação - *uma % alta de NC após este refinamento pode indicar muitos comentários realmente sem contexto/off-topic ou necessidade de mais ajustes*). Destaque algum tema específico que dominou a conversa, se for o caso.)

Instruções Adicionais:
*   Seja direto e focado em insights que possam gerar ações para o Itaú.
*   Baseie-se APENAS nos dados fornecidos no resumo. Não invente informações ou temas não listados.
*   Se algum dado crucial estiver faltando ou for insuficiente (ex: muito poucos comentários negativos para tirar conclusões sobre temas negativos), mencione essa limitação.
*   Mantenha a linguagem profissional e analítica.
*   Use bullet points (*) para listar os insights dentro de cada tópico.
"""


# --- Listas de Categorias Válidas ---
categorias_sentimento_validas = ["Positivo", "Negativo", "Neutro", "Não Classificado"]
categorias_tema_validas = [
    "Marca e Imagem",
    "Produtos e Serviços (Geral)",
    "Atendimento e Suporte",
    "Plataformas Digitais (App/Site/ATM)",
    "Figuras Públicas e Representantes",
    "Eventos e Campanhas Específicas",
    "Segurança e Fraude",
    "Solicitação/Dúvida/Sugestão (Transversal)", # Mantido aqui
    "Interação Social e Engajamento",
    "Não Classificado (Tema)"
]
categorias_erro = ["Erro Parsing", "Erro API"]
categorias_erro_tema_especifico = ["Erro API (Timeout)", "Erro API (Geral)", "Erro API (Modelo não iniciado)", "Erro API (Conteúdo Bloqueado)"]
todas_categorias_erro = list(set(categorias_erro + categorias_erro_tema_especifico))
categorias_excluir_sentimento = ["Não Classificado"] + todas_categorias_erro
# Excluir Interação Social dos gráficos/insights principais continua fazendo sentido
categorias_excluir_tema = ["Não Classificado (Tema)", "Interação Social e Engajamento"] + todas_categorias_erro


# --- Função para Analisar um Comentário ---
def analisar_comentario(comentario, modelo_gemini):
    # ... (Lógica interna da função permanece a mesma) ...
    if not comentario or not isinstance(comentario, str) or comentario.strip() == "":
        return "Não Classificado", "Não Classificado (Tema)"
    if not modelo_gemini:
        return "Erro API", "Erro API (Modelo não iniciado)"

    prompt_com_comentario = seu_prompt_completo.format(comment=comentario)
    try:
        safety_settings = {
            "HARM_CATEGORY_HARASSMENT": "BLOCK_NONE",
            "HARM_CATEGORY_HATE_SPEECH": "BLOCK_NONE",
            "HARM_CATEGORY_SEXUALLY_EXPLICIT": "BLOCK_NONE",
            "HARM_CATEGORY_DANGEROUS_CONTENT": "BLOCK_NONE"
        }
        request_options = {"timeout": 60}
        response = modelo_gemini.generate_content(
            prompt_com_comentario,
            safety_settings=safety_settings,
            request_options=request_options
        )
        texto_resposta = response.text.strip()
        sentimento_extraido = "Erro Parsing"
        tema_extraido = "Erro Parsing"
        linhas = texto_resposta.split('\n')

        for linha in linhas:
            linha_strip = linha.strip()
            if linha_strip.lower().startswith("sentimento:"):
                sentimento_extraido = linha_strip.split(":", 1)[1].strip()
            elif linha_strip.lower().startswith("tema:"):
                tema_extraido = linha_strip.split(":", 1)[1].strip()

        if sentimento_extraido == "Erro Parsing" or tema_extraido == "Erro Parsing":
             return "Erro Parsing", "Erro Parsing"
        if sentimento_extraido not in categorias_sentimento_validas:
             return "Erro Parsing", "Erro Parsing"

        if sentimento_extraido == "Não Classificado":
            return "Não Classificado", "Não Classificado (Tema)"
        else:
             if tema_extraido not in categorias_tema_validas or tema_extraido == "Não Classificado (Tema)":
                  return sentimento_extraido, "Erro Parsing"
             else:
                  return sentimento_extraido, tema_extraido

    except genai.types.StopCandidateException as e:
        return "Erro API", "Erro API (Conteúdo Bloqueado)"
    except Exception as e:
        error_type = "Erro API (Geral)"
        error_message = str(e).lower()
        if "timeout" in error_message or "deadline exceeded" in error_message:
            error_type = "Erro API (Timeout)"
        return "Erro API", error_type


# --- Função para Gerar Insights ---
# ... (Lógica interna da função permanece a mesma) ...
def gerar_insights(df_resultados_func, modelo_gemini):
    if df_resultados_func is None or df_resultados_func.empty:
        return "Não há dados suficientes para gerar insights."
    if not modelo_gemini:
        return "*Erro: Modelo Gemini não inicializado. Não é possível gerar insights.*"
    try:
        total_analisados_func = len(df_resultados_func)
        sent_counts_total = df_resultados_func['Sentimento_Classificado'].value_counts()
        count_pos_func = sent_counts_total.get('Positivo', 0)
        count_neg_func = sent_counts_total.get('Negativo', 0)
        count_neu_func = sent_counts_total.get('Neutro', 0)
        count_nc_err_func = total_analisados_func - (count_pos_func + count_neg_func + count_neu_func)
        perc_pos_func = (count_pos_func / total_analisados_func * 100) if total_analisados_func > 0 else 0
        perc_neg_func = (count_neg_func / total_analisados_func * 100) if total_analisados_func > 0 else 0
        perc_neu_func = (count_neu_func / total_analisados_func * 100) if total_analisados_func > 0 else 0
        perc_nc_err_func = (count_nc_err_func / total_analisados_func * 100) if total_analisados_func > 0 else 0
        df_tema_insights = df_resultados_func[~df_resultados_func['Tema_Classificado'].isin(categorias_excluir_tema)].copy()
        tema_counts_insights = df_tema_insights['Tema_Classificado'].value_counts()
        top_temas_formatado_func = ""
        total_temas_insights_func = 0
        if not tema_counts_insights.empty:
            total_temas_insights_func = tema_counts_insights.sum()
            top_temas_formatado_func = "\n".join([ f"    - {tema}: {count} ({count / total_temas_insights_func * 100:.1f}%)" for tema, count in tema_counts_insights.head(5).items()]) if total_temas_insights_func > 0 else "    - Nenhum tema relevante classificado."
        else: top_temas_formatado_func = "    - Nenhum tema relevante classificado."
        df_negativos = df_resultados_func[df_resultados_func['Sentimento_Classificado'] == 'Negativo'].copy()
        df_negativos_filtrados = df_negativos[~df_negativos['Tema_Classificado'].isin(categorias_excluir_tema)]
        tema_neg_counts = df_negativos_filtrados['Tema_Classificado'].value_counts()
        top_temas_negativos_formatado_func = ""
        total_temas_neg_func = 0
        if not tema_neg_counts.empty:
             total_temas_neg_func = tema_neg_counts.sum()
             top_temas_negativos_formatado_func = "\n".join([ f"    - {tema}: {count} ({count / total_temas_neg_func * 100:.1f}%)" for tema, count in tema_neg_counts.head(3).items()]) if total_temas_neg_func > 0 else "    - Nenhum tema negativo relevante classificado."
        else: top_temas_negativos_formatado_func = "    - Nenhum tema negativo relevante classificado (ou nenhum comentário negativo com tema válido)."
        prompt_final_insights = prompt_geracao_insights.format(total_comentarios_analisados=total_analisados_func, count_pos=count_pos_func, perc_pos=perc_pos_func, count_neg=count_neg_func, perc_neg=perc_neg_func, count_neu=count_neu_func, perc_neu=perc_neu_func, count_nc_err=count_nc_err_func, perc_nc_err=perc_nc_err_func, total_temas_insights=total_temas_insights_func, top_temas_formatado=top_temas_formatado_func, total_temas_neg=total_temas_neg_func, top_temas_negativos_formatado=top_temas_negativos_formatado_func)
        safety_settings = { "HARM_CATEGORY_HARASSMENT": "BLOCK_NONE", "HARM_CATEGORY_HATE_SPEECH": "BLOCK_NONE", "HARM_CATEGORY_SEXUALLY_EXPLICIT": "BLOCK_NONE", "HARM_CATEGORY_DANGEROUS_CONTENT": "BLOCK_NONE"}
        request_options = {"timeout": 90}
        response_insights = modelo_gemini.generate_content(prompt_final_insights, safety_settings=safety_settings, request_options=request_options)
        if response_insights and hasattr(response_insights, 'text'): st.session_state.insights_generated = response_insights.text.strip(); return st.session_state.insights_generated
        else:
            error_info = "Resposta da API vazia ou inválida.";
            if response_insights and hasattr(response_insights, 'prompt_feedback'): error_info = f"Possível bloqueio pela API. Feedback: {response_insights.prompt_feedback}"
            st.warning(f"Não foi possível gerar insights: {error_info}", icon="⚠️"); st.session_state.insights_generated = f"*Não foi possível gerar insights: {error_info}*"; return st.session_state.insights_generated
    except Exception as e: st.error(f"Erro durante a geração de insights: {e}", icon="🚨"); st.session_state.insights_generated = f"*Ocorreu um erro inesperado durante a geração dos insights: {str(e)}*"; return st.session_state.insights_generated


# --- Interface Principal ---
# ... (Restante da interface: Título, Controles, Pré-visualização, Análise, Resultados, Tabelas, Download, Insights) ...
# ... (Nenhuma mudança necessária no código da interface em si) ...

st.title("📊 Aplicativo para análise de sentimento e temática automatizado por IA")
st.markdown("""
Este aplicativo utiliza a IA Generativa do Google (Gemini) para classificar e analisar automaticamente o **sentimento**, **temática** e gerar insights dos comentários.
Desenvolvido pelo time de Social Intelligence do Hub de Inovação da iHouse/Oliver para o Itaú.
""")
st.markdown("---")

# --- Controles na Barra Lateral ---
st.sidebar.divider()
st.sidebar.header("Controles")
uploaded_file = st.sidebar.file_uploader(
    "1. Escolha o arquivo (.csv ou .xlsx)", type=["csv", "xlsx"], key="file_uploader",
    help="Faça upload de um arquivo CSV ou Excel que contenha uma coluna chamada 'conteúdo' com os textos a serem analisados."
)
coluna_conteudo = 'conteúdo'
botao_habilitado = st.session_state.get('api_key_configured', False) and uploaded_file is not None
analisar_btn = st.sidebar.button( "2. Analisar Comentários", key="analyze_button", disabled=(not botao_habilitado), help="Clique para iniciar a análise dos comentários na coluna 'conteúdo' do arquivo carregado.")
if not st.session_state.get('api_key_configured', False): st.sidebar.warning("API Key do Google não configurada ou inválida.", icon="⚠️")
if not uploaded_file: st.sidebar.info("Aguardando upload do arquivo...", icon="📤")
if botao_habilitado: st.sidebar.info("Pronto para analisar!", icon="✅")

# --- Área Principal: Pré-visualização e Resultados ---
df_original = None; df_para_analise = None; total_comentarios_para_analisar = 0
if uploaded_file is not None:
    try:
        if uploaded_file.name.endswith('.csv'):
            try: df_original = pd.read_csv(uploaded_file)
            except UnicodeDecodeError: uploaded_file.seek(0); df_original = pd.read_csv(uploaded_file, encoding='latin1')
        else: df_original = pd.read_excel(uploaded_file)
        if coluna_conteudo not in df_original.columns:
            st.error(f"Erro Crítico: Coluna '{coluna_conteudo}' não encontrada no arquivo '{uploaded_file.name}'. Verifique o nome da coluna.", icon="🚨"); df_original = None
        else:
            df_para_analise = df_original.copy(); df_para_analise.dropna(subset=[coluna_conteudo], inplace=True); df_para_analise = df_para_analise[df_para_analise[coluna_conteudo].astype(str).str.strip() != '']; total_comentarios_para_analisar = len(df_para_analise)
            st.subheader("Pré-visualização dos dados originais:")
            st.dataframe(df_original.head(10), use_container_width=True)
            if total_comentarios_para_analisar < len(df_original): st.info(f"Total de linhas no arquivo: {len(df_original)}. Total de comentários válidos para análise (coluna '{coluna_conteudo}' não vazia): **{total_comentarios_para_analisar}**", icon="ℹ️")
            else: st.info(f"Total de comentários válidos para análise: **{total_comentarios_para_analisar}**", icon="ℹ️")
    except Exception as e: st.error(f"Erro ao ler ou processar o arquivo '{uploaded_file.name}': {e}", icon="🚨"); df_original = None; df_para_analise = None

results_container = st.container()

# --- Lógica de Análise ---
if analisar_btn and df_para_analise is not None:
    if total_comentarios_para_analisar == 0: st.warning("Nenhum comentário válido encontrado na coluna '{coluna_conteudo}' para análise.", icon="⚠️")
    elif not model: st.error("Erro: Modelo Gemini não inicializado. Verifique a configuração da API Key na barra lateral.", icon="🚨")
    else:
        st.session_state.analysis_done = False; st.session_state.df_results = None; st.session_state.insights_generated = None
        with st.spinner(f"Analisando {total_comentarios_para_analisar} comentários... Isso pode levar alguns minutos."):
            progress_bar = st.progress(0.0); status_text = st.empty(); resultados_sentimento = []; resultados_tema = []; df_copy_analise = df_para_analise.copy(); start_time = time.time()
            for i, comentario in enumerate(df_copy_analise[coluna_conteudo]):
                sentimento, tema = analisar_comentario(str(comentario), model); resultados_sentimento.append(sentimento); resultados_tema.append(tema); progresso = (i + 1) / total_comentarios_para_analisar; progress_bar.progress(progresso); status_text.text(f"Analisando: {i+1}/{total_comentarios_para_analisar} ({progresso:.1%})")
            end_time = time.time(); tempo_total = end_time - start_time; progress_bar.empty(); status_text.success(f"✅ Análise concluída em {tempo_total:.2f} segundos!", icon="🎉")
            df_copy_analise['Sentimento_Classificado'] = resultados_sentimento; df_copy_analise['Tema_Classificado'] = resultados_tema; st.session_state.df_results = df_copy_analise; st.session_state.analysis_done = True

# --- Exibição dos Resultados ---
if st.session_state.analysis_done and st.session_state.df_results is not None:
    with results_container:
        df_results = st.session_state.df_results; total_analisados_results = len(df_results); st.markdown("---"); st.subheader("Visualização dos Resultados")
        df_sent_chart = df_results[~df_results['Sentimento_Classificado'].isin(categorias_excluir_sentimento)].copy(); sent_counts_chart = df_sent_chart['Sentimento_Classificado'].value_counts(); total_sent_chart = sent_counts_chart.sum(); nps_score_num = None
        if total_sent_chart > 0: count_pos_chart = sent_counts_chart.get('Positivo', 0); count_neg_chart = sent_counts_chart.get('Negativo', 0); perc_pos_chart = count_pos_chart / total_sent_chart; perc_neg_chart = count_neg_chart / total_sent_chart; nps_formula_standard = ((perc_pos_chart - perc_neg_chart) + 1) / 2 * 10; nps_score_num = max(0, min(10, nps_formula_standard))
        nps_col, chart_col1, chart_col2 = st.columns([1, 2, 2])
        with nps_col: st.markdown("##### NPS Social"); st.metric(label="(Escala 0-10)", value=f"{nps_score_num:.1f}" if nps_score_num is not None else "N/A"); st.caption("Sem dados P/N/Neu." if nps_score_num is None else "")
        with chart_col1: st.markdown("##### Distribuição de Sentimento"); # ...(código do gráfico de pizza igual)...
        with chart_col2: st.markdown("##### Distribuição Temática"); # ...(código do gráfico de barras igual)...
        # (Código dos gráficos e tabelas continua aqui, sem alterações necessárias)
        # ... (Cole o código dos gráficos e tabelas da versão anterior aqui) ...
        # --- Copie e cole o código de NPS, Gráficos, Tabelas e Download da versão anterior aqui ---
        # --- (O código abaixo é o final, incluindo a chamada de insights) ---
        with chart_col1:
            # st.markdown("##### Distribuição de Sentimento") # Já existe
            if total_sent_chart > 0:
                df_plot_sent = pd.DataFrame({'Sentimento': sent_counts_chart.index, 'Volume': sent_counts_chart.values})
                df_plot_sent['Sentimento'] = pd.Categorical(df_plot_sent['Sentimento'], categories=["Positivo", "Neutro", "Negativo"], ordered=True)
                df_plot_sent = df_plot_sent.sort_values('Sentimento')
                fig_sent = px.pie(df_plot_sent, names='Sentimento', values='Volume', hole=0.4, color='Sentimento', color_discrete_map={'Positivo': '#28a745', 'Negativo': '#dc3545', 'Neutro': '#ffc107'}, title='Sentimentos (Excluindo Não Classif./Erros)')
                fig_sent.update_traces(textposition='outside', textinfo='percent+label', hovertemplate="<b>%{label}</b><br>Volume: %{value}<br>Percentual: %{percent:.1%}<extra></extra>")
                fig_sent.update_layout(showlegend=False, title_x=0.5, height=350, margin=dict(l=10, r=10, t=40, b=10))
                st.plotly_chart(fig_sent, use_container_width=True)
            else: st.warning("Nenhum sentimento Positivo, Negativo ou Neutro classificado para exibir gráfico.", icon="📊")
        with chart_col2:
            # st.markdown("##### Distribuição Temática") # Já existe
            df_tema_chart = df_results[~df_results['Tema_Classificado'].isin(categorias_excluir_tema)].copy()
            tema_counts_chart = df_tema_chart['Tema_Classificado'].value_counts()
            total_tema_chart = tema_counts_chart.sum()
            if total_tema_chart > 0:
                tema_perc_chart = (tema_counts_chart / total_tema_chart * 100)
                df_plot_tema = pd.DataFrame({'Tema': tema_counts_chart.index, 'Volume': tema_counts_chart.values, 'Percentual': tema_perc_chart.values}).sort_values(by='Volume', ascending=False)
                fig_tema = px.bar(df_plot_tema, x='Tema', y='Volume', color_discrete_sequence=['#007bff']*len(df_plot_tema), title='Principais Temas (Excluindo NC/Erro/Interação)', hover_data={'Tema': False, 'Volume': True, 'Percentual': ':.1f%'}, text='Volume')
                fig_tema.update_traces(textposition='outside')
                fig_tema.update_layout(xaxis_title=None, yaxis_title="Volume Bruto", title_x=0.5, height=350, margin=dict(l=10, r=10, t=40, b=10))
                fig_tema.update_xaxes(tickangle= -30)
                st.plotly_chart(fig_tema, use_container_width=True)
            else: st.warning("Nenhum tema válido (excluindo NC/Erro/Interação) classificado para exibir gráfico.", icon="📊")

        st.markdown("---"); st.subheader("Tabelas de Resumo Completas"); col_t1, col_t2 = st.columns(2)
        with col_t1:
            st.markdown("###### Tabela 1: Sentimento (Completa)")
            todas_cats_sent = categorias_sentimento_validas + sorted(list(set(todas_categorias_erro))); sent_counts_all = df_results['Sentimento_Classificado'].value_counts().reindex(todas_cats_sent, fill_value=0); sent_perc_all = (sent_counts_all / total_analisados_results * 100) if total_analisados_results > 0 else 0; tabela_sent = pd.DataFrame({'Sentimento': sent_counts_all.index, 'Volume Bruto': sent_counts_all.values, 'Percentual (%)': sent_perc_all.values}); total_row_sent = pd.DataFrame({'Sentimento': ['Total Geral'], 'Volume Bruto': [total_analisados_results], 'Percentual (%)': [100.0]}); tabela_sent_final = pd.concat([tabela_sent[tabela_sent['Volume Bruto'] > 0], total_row_sent], ignore_index=True); st.table(tabela_sent_final.style.format({'Percentual (%)': '{:.2f}%'}))
        with col_t2:
            st.markdown("###### Tabela 2: Temática (Completa)")
            todas_cats_tema = categorias_tema_validas + sorted(list(set(todas_categorias_erro))); tema_counts_all = df_results['Tema_Classificado'].value_counts().reindex(todas_cats_tema, fill_value=0); tema_counts_all = tema_counts_all[~tema_counts_all.index.duplicated(keep='first')]; tema_perc_all = (tema_counts_all / total_analisados_results * 100) if total_analisados_results > 0 else 0; tabela_tema = pd.DataFrame({'Tema': tema_counts_all.index, 'Volume Bruto': tema_counts_all.values, 'Percentual (%)': tema_perc_all.values}); total_row_tema = pd.DataFrame({'Tema': ['Total Geral'], 'Volume Bruto': [total_analisados_results], 'Percentual (%)': [100.0]}); tabela_tema_final = pd.concat([tabela_tema[tabela_tema['Volume Bruto'] > 0], total_row_tema], ignore_index=True); st.table(tabela_tema_final.style.format({'Percentual (%)': '{:.2f}%'}))

        st.markdown("---"); st.subheader("Resultados Completos Detalhados"); st.dataframe(df_results, use_container_width=True)
        @st.cache_data
        def convert_df_to_csv(df_conv): return df_conv.to_csv(index=False, encoding='utf-8-sig').encode('utf-8-sig')
        if uploaded_file: base_name = uploaded_file.name.split('.')[0]; download_filename = f"{base_name}_analise_gemini.csv"
        else: download_filename = 'analise_gemini_resultados.csv'
        csv_output = convert_df_to_csv(df_results)
        st.download_button(label="💾 Download Resultados Completos (.csv)", data=csv_output, file_name=download_filename, mime='text/csv', key='download_csv', help="Baixa a tabela completa acima, incluindo as classificações de Sentimento e Tema, em formato CSV.")

        st.markdown("---"); st.subheader("💡 Insights e Percepções Acionáveis")
        if st.session_state.analysis_done and st.session_state.df_results is not None and model:
            if st.session_state.insights_generated is None:
                with st.spinner("Gerando insights com base nos resultados..."): gerar_insights(st.session_state.df_results, model)
            if st.session_state.insights_generated: st.markdown(st.session_state.insights_generated)
            else: st.warning("Não foi possível gerar ou carregar os insights.", icon="⚠️")
        elif not model: st.warning("Modelo Gemini não inicializado. Não é possível gerar insights.", icon="⚠️")
        else: st.info("Realize uma análise primeiro para poder gerar os insights.", icon="ℹ️")

elif not uploaded_file and not st.session_state.analysis_done :
     st.info("⬅️ Para começar, configure sua API Key (se necessário) e faça o upload de um arquivo .csv ou .xlsx na barra lateral.", icon="👈")