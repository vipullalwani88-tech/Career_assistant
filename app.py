import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
import os

# chroma_db folder is already in repo

st.set_page_config(page_title="Career Assistant", page_icon="🤖")
st.title("🤖 Personal Career Assistant")
st.caption("Powered by your study documents + Gemini AI")

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")

@st.cache_resource
def load_rag_chain():
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001",
        google_api_key=GEMINI_API_KEY
    )
    vectorstore = Chroma(
        persist_directory="chroma_db",
        embedding_function=embeddings,
        collection_name="career_assistant"
    )
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        google_api_key=GEMINI_API_KEY,
        temperature=0.3
    )
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
    template = """Use the following context to answer the question.
If you do not know the answer from the context, say you dont know.

Context: {context}

Question: {question}

Answer:"""
    prompt = PromptTemplate.from_template(template)
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return rag_chain

rag_chain = load_rag_chain()

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

if prompt := st.chat_input("Ask me anything from your documents..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            answer = rag_chain.invoke(prompt)
            st.write(answer)
    st.session_state.messages.append({"role": "assistant", "content": answer})
