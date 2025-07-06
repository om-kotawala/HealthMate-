from flask import Flask, request, jsonify
import os
import re
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv, find_dotenv
from huggingface_hub import InferenceClient
from transformers import pipeline

# Initialize Flask App
app = Flask(__name__)

# Disable TensorFlow oneDNN optimizations
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

# Load environment variables
load_dotenv(find_dotenv())

# Hugging Face API Token
HF_TOKEN = os.getenv("HF_TOKEN")
HUGGINGFACE_REPO_ID = "mistralai/Mistral-7B-Instruct-v0.3"

if not HF_TOKEN:
    raise ValueError("HF_TOKEN not found in environment variables. Please set it in your .env file.")

# Zero-Shot Classification Model for Intent Detection
classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
MEDICAL_CATEGORIES = ["medical", "health", "disease", "symptoms", "treatment", "medicine", "doctor"]

def is_medical_query(query: str) -> bool:
    """Check if the query is related to medical topics."""
    result = classifier(query, candidate_labels=MEDICAL_CATEGORIES)
    return result["labels"][0] in MEDICAL_CATEGORIES and result["scores"][0] > 0.7

# Hugging Face Inference Wrapper
class HuggingFaceInferenceWrapper:
    def __init__(self, model_id: str, token: str):
        self.client = InferenceClient(model=model_id, token=token)

    def invoke(self, prompt: str, temperature=0.5, max_new_tokens=512) -> str:
        response = self.client.text_generation(
            prompt,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
        )
        return response.strip()

# Initialize LLM
llm = HuggingFaceInferenceWrapper(model_id=HUGGINGFACE_REPO_ID, token=HF_TOKEN)

# Custom Prompt Template
custom_prompt_template = PromptTemplate(
    template="""Use the provided context to answer the query. If you don't know the answer, just say you don't know. Do not make up an answer.

Context: {context}
Query: {query}

Provide a direct answer. If the response includes a medicine name, add the warning: 'I am not a doctor. Please consult a medical professional before taking any medicine.'
""",
    input_variables=["context", "query"],
)

# Load FAISS Vectorstore
DB_FAISS_PATH = "vectorstore/db_faiss"
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
try:
    db = FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)
except Exception as e:
    raise RuntimeError(f"Failed to load FAISS database: {e}")

# Define Document Formatting Function
def format_docs(inputs):
    context = "\n".join([doc.page_content for doc in inputs["context"]]) if inputs["context"] else "No relevant context found."
    return {"context": context, "query": inputs["query"]}

# Medicine Warning
def add_medicine_warning(response: str) -> str:
    medicine_pattern = re.compile(r"\b(aspirin|paracetamol|antibiotic|painkiller|metformin|dolo|cetirizine|ibuprofen|naproxen|amoxicillin)\b", re.IGNORECASE)
    if medicine_pattern.search(response):
        response += "\n⚠️ I am not a doctor. Please consult a medical professional before taking any medicine."
    return response

# Retrieval Chain
retriever = db.as_retriever(search_kwargs={"k": 3})
retrieval_chain = (
    RunnableParallel({"context": retriever, "query": RunnablePassthrough()})
    | RunnableLambda(format_docs)
    | (lambda inputs: custom_prompt_template.format(**inputs))
    | (lambda prompt: llm.invoke(prompt))
    | RunnableLambda(lambda response: add_medicine_warning(response))
)

# Health Check Route
@app.route("/", methods=["GET"])
def home():
    return jsonify({"message": "Healthmate API is running! Use /get_response to send queries."})

@app.route("/get_response", methods=["POST"])
def get_response():
    data = request.json
    query = data.get("query", "").strip()
    
    if not query:
        return jsonify({"response": "❌ Invalid input. Please provide a valid query."})

    if not is_medical_query(query):
        return jsonify({"response": "❌ Sorry, I can only answer medical-related queries."})

    try:
        response = retrieval_chain.invoke(query)
        return jsonify({"response": response})
    except Exception as e:
        return jsonify({"response": f"❌ An error occurred: {str(e)}"})

if __name__ == "__main__":
    app.run(debug=True)
