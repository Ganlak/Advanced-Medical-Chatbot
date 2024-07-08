from src.helper import load_pdf, text_split, download_hugging_face_embeddings
import warnings
from langchain_community.vectorstores import Chroma

# Ignore LangChainDeprecationWarning
warnings.filterwarnings("ignore", category=DeprecationWarning)

extracted_data = load_pdf(r"E:\LLM_Project\Medical-Chatbot\data")
text_chunks = text_split(extracted_data)
embeddings = download_hugging_face_embeddings()

#Persisting the DB
persist_directory = 'db'

#Creating Embeddings for Each of The Text Chunks & storing
vectordb = Chroma.from_documents(documents=text_chunks,embedding=embeddings,persist_directory=persist_directory)
