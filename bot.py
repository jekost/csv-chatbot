from langchain.vectorstores import Pinecone
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain.llms import OpenAI
import pinecone

#OpenAI
EMBEDDING_MODEL = "text-embedding-ada-002"
OPENAI_API_KEY=""
#Pinecone
PINECONE_API_KEY = ""
PINECONE_API_ENV = "us-west4-gcp-free"
index_name = ""

#Pinecone
pinecone.init(
    api_key=PINECONE_API_KEY,
    environment=PINECONE_API_ENV
)

#ligipääs embeddingutele
embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY,model=EMBEDDING_MODEL)
vectorstore = Pinecone.from_existing_index(index_name=index_name, embedding=embeddings)

#Langchaini teegiga loodud vestluskett
qa = ConversationalRetrievalChain.from_llm(
    llm=OpenAI(temperature=0.2, max_tokens=300,openai_api_key=OPENAI_API_KEY),
    retriever=vectorstore.as_retriever(),
    return_source_documents=True
)

#programmi töö, kus saab mudeliga suhelda

chat_history = []
while True:
    query = input("küsimus: ")

    if query == "":
        exit()

    #hidden prompt on n-ö mudeli "korrale kutsumine", et ta ikka teeks täpselt nii nagu me tahaks
    hidden_prompt = "Vasta küsimusele antud konteksti põhjal. "
    result = qa({"question": hidden_prompt+query, "chat_history": chat_history})
    chat_history.append((query, result["answer"]))

    print(result["answer"])
