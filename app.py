import os
import json
import datetime
import requests
from flask import Flask, request, jsonify, render_template
from firebase_admin import credentials, firestore, initialize_app
from PyPDF2 import PdfReader
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.docstore.document import Document
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.chat_models import ChatOpenAI
from langchain.chat_models import ChatOpenAI
os.environ["OPENAI_API_KEY"] = "sk-dummy-key"

# Load environment variables
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
IPINFO_TOKEN = os.getenv("IPINFO_TOKEN")

# Firebase setup
cred = credentials.Certificate("firebase_key.json")
initialize_app(cred)
db = firestore.client()

# Flask setup
app = Flask(__name__)

# Load and process PDFs 
documents = []
for filename in os.listdir("data"):
    if filename.endswith(".pdf"):
        pdf_reader = PdfReader(os.path.join("data", filename))
        raw_text = "\n".join(page.extract_text() for page in pdf_reader.pages if page.extract_text())
        documents.append(Document(page_content=raw_text))

text_splitter = CharacterTextSplitter(chunk_size=300, chunk_overlap=30)
docs = text_splitter.split_documents(documents)
print(f"Total documents after split: {len(docs)}")

# Embeddings and vectorstore
embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
vectorstore = Chroma.from_documents(docs, embedding=embeddings)
retriever = vectorstore.as_retriever()

# QA chain setup
llm = ChatOpenAI(base_url="https://openrouter.ai/api/v1", api_key=OPENROUTER_API_KEY, model="mistralai/mixtral-8x7b-instruct")
qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

known_products = {
    "IntelliFlo3 VSF Pump": "Variable Speed & Flow Pool Pump",
    "WhisperFlo VST": "Variable Speed Pool Pump",
    "SuperFlo VST": "Variable Speed Pool Pump",
    "WhisperFlo High Performance Pump": "Single Speed Pool Pump",
    "SuperFlo High Performance Pump": "Single Speed Pool Pump",
    "OptiFlo Aboveground Pool Pump": "Aboveground Pool Pump",
    "MasterTemp Heater": "Pool Heater",
    "MasterTemp HD": "Heavy Duty Pool Heater",
    "ETi 400": "High-Efficiency Pool Heater",
    "Max-E-Therm": "High Performance Pool and Spa Heater",
    "EasyTouch Control System": "Automation Control System",
    "IntelliCenter Control System": "Automation Control System",
    "FNS Plus Filter": "Pool Filter",
    "Whole House Carbon Filtration System": "Carbon Filtration System",
    "Salt Chlorinator": "Salt Chlorinator",
    "Water Softener": "Softener",
    "UV Sanitizer": "UV Pool Sanitizer",
    "Prowler 910": "Robotic Pool Cleaner (Aboveground)",
    "Prowler 920": "Robotic Pool Cleaner (Inground)",
    "Prowler 930W": "WiFi Robotic Pool Cleaner (Inground)",
    "Aurora Layne Verti-Line Series 1100": "Multi-Stage Vertical Turbine Pump"
}

# fallback keywords
fallback_keywords = {
    "carbon": "Whole House Carbon Filtration System",
    "filtration": "Carbon Filtration System",
    "softener": "Water Softener",
    "heater": "Pool Heater",
    "pump": "Pool Pump",
    "variable speed": "Variable Speed Pool Pump",
    "high efficiency": "High-Efficiency Pool Heater",
    "high performance": "High Performance Pool and Spa Heater",
    "chlorinator": "Salt Chlorinator",
    "control": "Automation Control System",
    "filter": "Pool Filter",
    "uv": "UV Sanitizer",
    "prowler": "Robotic Pool Cleaner",
    "robotic": "Robotic Pool Cleaner",
    "spa": "Pool and Spa Heater",
    "intelliflo": "Variable Speed & Flow Pool Pump",
    "whisperflo": "Pool Pump",
    "superflo": "Pool Pump",
    "intelicenter": "Automation Control System",
    "aurora": "Vertical Turbine Pump",
    "turbine": "Vertical Turbine Pump"
}


def detect_product_from_query(query, rag_response=None):
    query = query.lower()
    rag_response = rag_response.lower() if rag_response else ""

    for product_name, product_type in known_products.items():
        if product_name.lower() in query:
            return {"product_name": product_name, "product_type": product_type}

    for keyword, fallback_name in fallback_keywords.items():
        if keyword in query:
            return {"product_name": fallback_name, "product_type": known_products.get(fallback_name, fallback_name.split()[-1])}

    for product_name, product_type in known_products.items():
        if product_name.lower() in rag_response:
            return {"product_name": product_name, "product_type": product_type}

    return {"product_name": "Other", "product_type": "Unknown"}

def extract_products(query):

    query_lower = query.lower()
    matched = []
    for product_name, product_type in known_products.items():
        if product_name.lower() in query_lower:
            matched.append({"product_name": product_name, "product_type": product_type})
    return matched

def fetch_location():
    try:
        response = requests.get(f"https://ipinfo.io/json?token={IPINFO_TOKEN}", timeout=3)
        data = response.json()
        location = f"{data.get('city', '')}, {data.get('region', '')}, {data.get('country', '')}"
        return location.strip(', ')
    except Exception as e:
        print(f"Failed to fetch location: {e}")
        return "Unknown"

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json()
    user_query = data.get("query")
    email = data.get("email")
    session_id = data.get("session_id")

    user_ref = db.collection("users").document(email)
    session_ref = user_ref.collection("sessions").document(session_id)
    session_doc = session_ref.get()

    # If this session exists, check cache for repeated query
    if session_doc.exists:
        existing = session_doc.to_dict().get("messages", [])
        for m in existing:
            if m["query"].strip().lower() == user_query.strip().lower():
                return jsonify({"response": m["response"]})

    # Otherwise generate a new answer
    try:
        rag_response = qa_chain.run(user_query)
    except Exception as e:
        print(f"🔴 QA Chain Error: {e}")
        rag_response = {"content": "Something went wrong."}

    answer_text = rag_response.get("content") if isinstance(rag_response, dict) else rag_response
    if not isinstance(answer_text, str) or not answer_text.strip():
        answer_text = "Something went wrong."

    # Extract product info for this prompt
    products = extract_products(user_query)  
    prod_info = detect_product_from_query(user_query, answer_text)  

    # Build the full message entry
    message_entry = {
        "query": user_query,
        "response": answer_text,
        "product_name": prod_info["product_name"],
        "product_type": prod_info["product_type"],
        "products": products,
        "timestamp": datetime.datetime.utcnow().isoformat()
    }

    # Append to existing session or create a new one
    if session_doc.exists:
        session_ref.update({
            "messages": firestore.ArrayUnion([message_entry])
        })
    else:
        location = fetch_location()
        session_ref.set({
            "email": email,
            "session_id": session_id,
            "created_at": datetime.datetime.utcnow().isoformat(),
            "title": user_query[:60] + ("..." if len(user_query) > 60 else ""),
            "location": location,
            "messages": [message_entry]
        })
    return jsonify({"response": answer_text})


@app.route("/get_chat_titles", methods=["POST"])
def get_chat_titles():
    email = request.json.get("email")
    sessions = (
        db.collection("users")
          .document(email)
          .collection("sessions")
          .order_by("created_at", direction=firestore.Query.DESCENDING)
          .stream()
    )
    titles = []
    for s in sessions:
        data = s.to_dict()
        # Use stored title or fallback to first message query
        title = data.get("title") or (data.get("messages", [])[0]["query"] if data.get("messages") else "No Title")
        titles.append({"session_id": s.id, "title": title})
    return jsonify({"titles": titles})


@app.route("/get_chat_by_id", methods=["POST"])
def get_chat_by_id():
    email = request.json.get("email")
    session_id = request.json.get("chat_id")
    session_ref = (
        db.collection("users")
          .document(email)
          .collection("sessions")
          .document(session_id)
    )
    session_doc = session_ref.get()
    if session_doc.exists:
        # Return only messages so front-end can render full conversation
        return jsonify({"messages": session_doc.to_dict().get("messages", [])})
    return jsonify({"error": "Chat not found"}), 404


if __name__ == "__main__":
    app.run(debug=True)
