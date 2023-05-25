from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Pinecone
from langchain.embeddings.openai import OpenAIEmbeddings
import pinecone

#võtmed,keskkonnad,pinecone'i indexi nimi

#OpenAI
OPENAI_API_KEY=""
#Pinecone
PINECONE_API_KEY = ""
PINECONE_API_ENV = ""
index_name=""

#faili link arvutis
csv=""
pdf=""

#loader = PyPDFLoader(pdf)
#loader = CSVLoader(csv,encoding="utf-8")

document = loader.load()# Add the loaded document to our list
text_splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=50)

texts = text_splitter.split_documents(document)
print(f"chunked_annual_report length: {len(texts)}")

#Pinecone käivitamine
pinecone.init(
    api_key=PINECONE_API_KEY,
    environment=PINECONE_API_ENV
)

embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
#Reaalne tööosa, kus tükeldatud faili embedditakse ja laetakse üles
for text in texts:
    Pinecone.from_texts([text.page_content], embeddings, index_name=index_name)