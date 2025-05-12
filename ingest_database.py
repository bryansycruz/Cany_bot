from langchain_community.document_loaders import UnstructuredPDFLoader, PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_chroma import Chroma
from uuid import uuid4

# import the .env file
from dotenv import load_dotenv
load_dotenv()

#configuracion 
DATA_PATH = r"data"
CHROMA_PATH = r"chroma_db"

# modelo de embeddings
embeddings_model = OpenAIEmbeddings(model="text-embedding-3-large")

#  Vectorestore
vector_store = Chroma(
    collection_name="example_collection",
    embedding_function=embeddings_model,
    persist_directory=CHROMA_PATH,
)

# lCargar los documentos PDF
loader = PyPDFDirectoryLoader(DATA_PATH)

raw_documents = loader.load()

# Dividir los documentos en fragmentos

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=300,
    chunk_overlap=100,
    length_function=len,
    is_separator_regex=False,
)

# creacion de los chunks
chunks = text_splitter.split_documents(raw_documents)

# creacion de los  ID's
uuids = [str(uuid4()) for _ in range(len(chunks))]


# agregar los chunks en los vectores 
vector_store.add_documents(documents=chunks, ids=uuids)