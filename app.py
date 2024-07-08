from flask import Flask, render_template, jsonify, request
from src.helper import download_hugging_face_embeddings
from langchain_community.vectorstores import Chroma
from langchain.prompts import PromptTemplate
from langchain_community.llms import CTransformers
from langchain.chains import RetrievalQA
from transformers import AutoTokenizer, AutoModel
from src.prompt import *

app = Flask(__name__)

#download embedding model
embeddings = download_hugging_face_embeddings()

db_path = r"E:\LLM_Project\Medical-Chatbot\db"

#loading the vectordb
vectordb = Chroma(persist_directory=db_path,embedding_function = embeddings)

#Making retriever
retriever = vectordb.as_retriever(search_kwargs={"k": 2})

# Define the prompt template
PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

# Initialize the CTransformers model
llm = CTransformers(model="model/llama-2-7b-chat.ggmlv3.q4_0.bin",
                   model_type="llama",
                   config={'max_new_tokens': 512, 'temperature': 0.8})

# Initialize the RetrievalQA chain
qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",  # Adjust this based on your specific setup
    retriever=retriever,
    return_source_documents=True,
    chain_type_kwargs={"prompt": PROMPT}  # Pass the prompt template as kwargs
)


@app.route("/")
def index():
    return render_template('chat.html')



@app.route("/get", methods=["GET", "POST"])
def chat():
    msg = request.form["msg"]
    input = msg
    print(input)
    result=qa({"query": input})
    print("Response : ", result["result"])
    return str(result["result"])



if __name__ == '__main__':
    app.run(host="0.0.0.0", port= 8080, debug= True)


