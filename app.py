import os
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.prompts.prompt import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_community.document_loaders import Docx2txtLoader, UnstructuredExcelLoader, CSVLoader, TextLoader, PyPDFLoader
from prompt import template
from langchain_community.vectorstores import FAISS


class RAGAssistant:
    def __init__(self):
        self.load_env_variables()
        self.setup_prompt_template()
        self.retriever = None
        self.llm = ChatOpenAI(model="gpt-4", temperature=0)

    def load_env_variables(self):
        load_dotenv('var.env')
        self.openai_api_key = os.getenv('OPENAI_API_KEY')

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
            chunk_size=500, chunk_overlap=100)
        docs = text_splitter.split_documents(documents)
        embeddings = OpenAIEmbeddings()
        vectorstore = FAISS.from_documents(
            docs, embeddings)

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
