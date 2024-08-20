import os
import streamlit as st
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.prompts.prompt import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_community.document_loaders import Docx2txtLoader, UnstructuredExcelLoader, CSVLoader, TextLoader, PyPDFLoader
# from langchain_community.vectorstores import Chroma
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone
from prompt import template


class RAGAssistant:
    def __init__(self):
        self.load_env_variables()
        self.setup_prompt_template()
        self.retriever = None
        self.llm = ChatOpenAI(model="gpt-4o", temperature=0)

    def load_env_variables(self):
        load_dotenv('var.env')
        self.openai_api_key = os.getenv('OPENAI_API_KEY')
        self.pinecone_api_key = os.getenv("PINECONE_API_KEY")
        self.pinecone_index_name = os.getenv("PINECONE_INDEX_NAME")

    def setup_prompt_template(self):
        self.prompt_template = PromptTemplate(
            input_variables=["context", "question"],
            template=template

        )

    def process_document(self, file_path):
        if file_path.endswith('.pdf'):
            loader = PyPDFLoader(file_path)
        elif file_path.endswith('.txt'):
            loader = TextLoader(file_path)
        elif file_path.endswith('.csv'):
            loader = CSVLoader(file_path=file_path)
        elif file_path.endswith('.xlsx'):
            loader = UnstructuredExcelLoader(file_path, mode="elements")
        elif file_path.endswith('.docx'):
            loader = Docx2txtLoader(file_path)
        else:
            raise ValueError("Unsupported file type.")

        documents = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=200)
        docs = text_splitter.split_documents(documents)
        embeddings = OpenAIEmbeddings()
        Pinecone(api_key=self.pinecone_api_key, environment='us-east-1')
        vectorstore = PineconeVectorStore.from_documents(
            docs, embeddings, index_name=self.pinecone_index_name)

        self.retriever = vectorstore.as_retriever()

    def analyze_document(self, query="Analyze this document and provide a JSON summary of its key information."):
        chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type='stuff',
            retriever=self.retriever,
            chain_type_kwargs={"prompt": self.prompt_template}
        )
        response = chain.invoke(query)
        return response['result']


def main():
    st.set_page_config(page_title="Document Analyzer", layout="wide")
    st.title("Document Analyzer - JSON Output")

    rag_assistant = RAGAssistant()

    uploaded_file = st.file_uploader("Upload a document", type=[
                                     "txt", "pdf", "csv", "xlsx", "docx"])

    if uploaded_file is not None:
        with st.spinner("Processing document..."):
            # Save the uploaded file temporarily
            with open(uploaded_file.name, "wb") as f:
                f.write(uploaded_file.getbuffer())

            # Process the document
            rag_assistant.process_document(uploaded_file.name)

            # Analyze the document
            json_output = rag_assistant.analyze_document()

            # Remove the temporary file
            os.remove(uploaded_file.name)

        st.subheader("JSON Output:")
        st.json(json_output)


if __name__ == "__main__":
    main()
