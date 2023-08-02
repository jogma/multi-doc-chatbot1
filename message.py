import os

from dotenv import load_dotenv
from typing import List, Tuple

from fastapi import APIRouter, FastAPI
from fastapi.middleware.cors import CORSMiddleware
from langchain.document_loaders import (
    PyPDFLoader,
    Docx2txtLoader,
    TextLoader,
    WebBaseLoader,
)
from langchain.document_loaders.base import BaseLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.schema import Document
from langchain.text_splitter import CharacterTextSplitter

from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

load_dotenv(".env")


class URLsLoader(BaseLoader):
    _file_path: str

    def __init__(self, file_path: str):
        self._file_path = file_path

    def load(self) -> List[Document]:
        documents = []
        with open(self._file_path) as file:
            for line in file.readlines():
                url = line.strip()
                if url:
                    documents.extend(WebBaseLoader(url).load())
                    print(f"{url} loaded as a document")
        return documents


class GenericLoader(BaseLoader):
    def load(self) -> List[Document]:
        documents = []
        # Create a List of Documents from all of our files in the ./docs folder
        for file in os.listdir("docs"):
            if file.endswith(".pdf"):
                pdf_path = "./docs/" + file
                loader = PyPDFLoader(pdf_path)
                documents.extend(loader.load())
            elif file.endswith(".docx") or file.endswith(".doc"):
                doc_path = "./docs/" + file
                loader = Docx2txtLoader(doc_path)
                documents.extend(loader.load())
            elif file.endswith(".txt"):
                text_path = "./docs/" + file
                loader = TextLoader(text_path)
                documents.extend(loader.load())
            elif file.endswith(".urls"):
                urls_path = "./docs/" + file
                loader = URLsLoader(urls_path)
                documents.extend(loader.load())
        return documents


class ChainMaker:
    _loader: BaseLoader

    def __init__(self, loader: BaseLoader):
        self._loader = loader

    def make(self) -> ConversationalRetrievalChain:
        documents = self._loader.load()

        # Split the documents into smaller chunks
        text_splitter = CharacterTextSplitter(
            chunk_size=2048, chunk_overlap=10
        )
        documents = text_splitter.split_documents(documents)

        # Convert the document chunks to embedding and save them to the vector
        # store
        vectordb = Chroma.from_documents(
            documents, embedding=OpenAIEmbeddings(), persist_directory="./data"
        )
        vectordb.persist()

        # create and return our Q&A chain
        return ConversationalRetrievalChain.from_llm(
            ChatOpenAI(temperature=0.7, model_name="gpt-3.5-turbo"),
            retriever=vectordb.as_retriever(search_kwargs={"k": 6}),
            return_source_documents=True,
            verbose=False,
        )


class Orchestrator(FileSystemEventHandler):
    _chain_maker: ChainMaker
    _chat_history: List[Tuple[str, str]]

    def __init__(self):
        self._chain_maker = ChainMaker(GenericLoader())
        self._chain = self._chain_maker.make()
        self._chat_history = []

        router = APIRouter()
        router.add_api_route("/answer", self._get_answer, methods=["GET"])
        app = FastAPI()
        app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        app.include_router(router)

    async def _get_answer(self, prompt: str):
        print(f"Getting answer for {repr(prompt)}")
        result = self._chain(
            {"question": prompt, "chat_history": self._chat_history}
        )
        self._chat_history.append((prompt, result["answer"]))
        print(f"The answer is {repr(result['answer'])}")
        return result["answer"]

    def on_modified(self, event):
        print(event.src_path, "modified.")
        self._chain = self._chain_maker.make()


observer = Observer()
observer.schedule(Orchestrator(), "./docs/", recursive=False)
observer.start()
try:
    while observer.is_alive():
        observer.join(1)
except KeyboardInterrupt:
    observer.stop()
observer.join()
