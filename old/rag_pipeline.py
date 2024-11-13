from langchain_community.vectorstores import Chroma
from langchain_community.chat_models import ChatOllama
from langchain_community.embeddings import FastEmbedEmbeddings
from langchain.schema.output_parser import StrOutputParser
from langchain_community.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema.runnable import RunnablePassthrough
from langchain.prompts import PromptTemplate
from langchain_community.vectorstores.utils import filter_complex_metadata


class ChatPDF:
    vector_store = None
    retriever = None
    chain = None

    def __init__(self):
        self.model = ChatOllama(model="llama3.2")
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=100)
        self.prompt = PromptTemplate.from_template(
            """
            <s> [INST] Vous êtes un assistant pour les tâches de réponse aux questions. Utilisez les éléments de contexte suivants pour répondre à la question. 
            Si vous ne connaissez pas la réponse, dites simplement que vous ne savez pas.. Utilisez trois phrases
             maximum et soyez concis dans votre réponse. [/INST] </s> 
            [INST] Question: {question} 
            Context: {context} 
            Answer: [/INST]
            """
        )

    """
    Ingest:
    DirectoryLoader is used to load the log files uploaded by the user.
    RecursiveCharacterSplitter (from LangChain) splits log files into smaller chunks.
    filter_complex_metadata filters out complex metadata not supported by ChromaDB.
    Chroma is used for vector storage and Qdrant FastEmbed is the embedding model.
    The lightweight model is transformed into a retriever with a score threshold of 0.5 and k=3 (returns top 3 chunks with highest scores above 0.5)
    LCEL constructs conversation chain.
    """
    def ingest(self, pdf_file_path: str):
        docs = DirectoryLoader("./data", glob="**/*.PRD")
        chunks = self.text_splitter.split_documents(docs)
        chunks = filter_complex_metadata(chunks)

        vector_store = Chroma.from_documents(documents=chunks, embedding=FastEmbedEmbeddings())
        self.retriever = vector_store.as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs={
                "k": 3,
                "score_threshold": 0.5,
            },
        )

        self.chain = ({"context": self.retriever, "question": RunnablePassthrough()}
                      | self.prompt
                      | self.model
                      | StrOutputParser())

    """
    Ask:
    Passes user's question into our predefined chain and returns results
    """
    def ask(self, query: str):
        if not self.chain:
            return "Please, add a PDF document first."

        return self.chain.invoke(query)

    """
    Clear:
    Clears previous chat session and storage when a new PDF file is uploaded.
    """
    def clear(self):
        self.vector_store = None
        self.retriever = None
        self.chain = None