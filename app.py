import streamlit as st
import os
import tempfile
from langchain_groq import ChatGroq
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain import hub

# --- CONFIGURAÇÃO DA PÁGINA (Layout otimizado para mobile) ---
st.set_page_config(page_title="AI Mobile Doc", layout="centered")
st.header("📱 Assistente de PDF")

# --- GERENCIAMENTO DE CHAVE (Secrets ou Input) ---
# Tenta pegar a chave dos Secrets do Streamlit Cloud primeiro
groq_key = st.secrets.get("GROQ_API_KEY", "")

with st.sidebar:
    st.title("Configurações")
    # Se a chave não estiver nos Secrets, permite digitar manualmente
    if not groq_key:
        groq_key = st.text_input("Cole sua Groq API Key:", type="password")
    else:
        st.success("Chave carregada via Secrets! ✅")
    
    modelo_selecionado = st.selectbox("Modelo:", ["llama3-8b-8192", "mixtral-8x7b-32768"])
    st.divider()
    if st.button("Limpar Histórico"):
        st.session_state.messages = []
        st.rerun()

# --- MOTOR DE PROCESSAMENTO ---
@st.cache_resource
def processar_documento(conteudo_arquivo):
    with tempfile.NamedTemporaryFile(delete=False) as tf:
        tf.write(conteudo_arquivo)
        caminho_temp = tf.name
    
    loader = PyPDFLoader(caminho_temp)
    documentos = loader.load()
    
    # Divisão otimizada para modelos menores
    splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=100)
    chunks = splitter.split_documents(documentos)
    
    # Embeddings grátis que rodam no servidor (sem custo)
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    vector_store = Chroma.from_documents(documents=chunks, embedding=embeddings)
    os.remove(caminho_temp)
    return vector_store.as_retriever(search_kwargs={"k": 3})

# --- INTERFACE DE UPLOAD ---
arquivo = st.file_uploader("Escolha um manual em PDF", type="pdf")

if arquivo and groq_key:
    # Indexação do PDF
    if "meu_retriever" not in st.session_state:
        with st.spinner("Lendo documento..."):
            st.session_state.meu_retriever = processar_documento(arquivo.getvalue())
            st.toast("PDF pronto para conversa!", icon="🚀")

    # --- CHAT ---
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Exibe as mensagens com estilo de chat
    for m in st.session_state.messages:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])

    # Entrada do usuário
    if pergunta := st.chat_input("Diga sua dúvida..."):
        st.session_state.messages.append({"role": "user", "content": pergunta})
        with st.chat_message("user"):
            st.markdown(pergunta)

        # Preparação da IA (Groq)
        llm = ChatGroq(groq_api_key=groq_key, model_name=modelo_selecionado)
        prompt = hub.pull("rlm/rag-prompt")

        chain = (
            {"context": st.session_state.meu_retriever, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
        )

        # Resposta da IA
        with st.chat_message("assistant"):
            with st.spinner("Pensando..."):
                resposta = chain.invoke(pergunta)
                st.markdown(resposta)
                st.session_state.messages.append({"role": "assistant", "content": resposta})

elif not groq_key:
    st.info("💡 Dica: Configure sua GROQ_API_KEY na barra lateral para liberar o chat.")