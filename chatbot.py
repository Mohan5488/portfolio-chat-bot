import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
from langchain.chains import ConversationalRetrievalChain
from langchain_community.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain_groq import ChatGroq
from dotenv import load_dotenv
from langchain_community.embeddings import OpenAIEmbeddings
load_dotenv()

embedding = OpenAIEmbeddings()


llm = ChatGroq(model="openai/gpt-oss-120b")

vector_store = FAISS.load_local(
    "vectorstore-openai",
    embedding,
    allow_dangerous_deserialization=True
)

retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 3})
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

qa = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=retriever,
    memory=memory
)

def ask_question(question):
    result = qa.invoke({"question": question})
    return result["answer"]
