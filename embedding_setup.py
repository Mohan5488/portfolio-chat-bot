from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

docs = []
for file in [
    "data/1_personal_info.txt", "data/2_about_me.txt", "data/3_education.txt",
    "data/4_skills.txt", "data/5_experience.txt", "data/6_projects.txt",
    "data/7_certifications.txt", "data/8_faqs.txt"
]:
    loader = TextLoader(file)
    docs.extend(loader.load())

text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
chunks = text_splitter.split_documents(docs)

embedding = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vector_store = FAISS.from_documents(chunks, embedding)

vector_store.save_local("vectorstore")
