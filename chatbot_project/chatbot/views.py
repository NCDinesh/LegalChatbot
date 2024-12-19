import os
import time
import warnings
import requests
from django.shortcuts import render
from django.http import JsonResponse
from PyPDF2 import PdfReader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import Chroma
import joblib
from dotenv import load_dotenv

# Load the .env file
load_dotenv()

# Access Hugging Face API token and model URL
API_TOKEN = os.getenv("HF_API_TOKEN")
API_URL = "https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.3"
warnings.filterwarnings("ignore")

# Define the prompt template
template = """
<s>[INST] <<SYS>>
You are a helpful AI assistant who has to act as an advocate in the case of country Nepal, not others.
Answer based on the context provided. Don't answer unnecessarily if you don't find the context.
<</SYS>>
{context}
Question: {question}
Helpful Answer: [/INST]
"""
prompt = PromptTemplate.from_template(template)


def query_huggingface(payload):
    """Send a query to Hugging Face Inference API and return the response."""
    headers = {"Authorization": f"Bearer {API_TOKEN}"}
    response = requests.post(API_URL, headers=headers, json=payload)
    return response.json()


def process_pdf():
    """Load and process the PDF to extract text."""
    reader = PdfReader("data/fine_tune_data.pdf")
    raw_text = ""
    for i, page in enumerate(reader.pages):
        text = page.extract_text()
        if text:
            raw_text += text
    return raw_text


def get_vectorstore(texts):
    """Load or create the vectorstore."""
    embeddings_file = "./data/genderviolence.joblib"
    if os.path.exists(embeddings_file):
        embeddings = joblib.load(embeddings_file)
    else:
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        joblib.dump(embeddings, embeddings_file)

    vectorstore = Chroma.from_texts(texts, embeddings, persist_directory=None)
    return vectorstore.as_retriever()


def chatbot_view(request):
    """Render chatbot interface and handle user queries."""
    if request.method == "POST":
        user_query = request.POST.get("query")
        if not user_query:
            return JsonResponse({"response": "Please enter a valid query."})

        # Process the PDF and split into chunks
        raw_text = process_pdf()
        text_splitter = CharacterTextSplitter(
            separator="\n",
            chunk_size=350,
            chunk_overlap=20,
            length_function=len,
        )
        texts = text_splitter.split_text(raw_text)

        # Retrieve relevant documents
        retriever = get_vectorstore(texts)
        documents = retriever.get_relevant_documents(user_query)
        context = "\n".join([doc.page_content for doc in documents])

        # Format the prompt
        full_prompt = template.format(context=context, question=user_query)

        # Create the payload
        config = {
            "max_new_tokens": 500,
            "temperature": 0.7,
            "context_length": 2048,
        }
        payload = {"inputs": full_prompt, "parameters": config}

        try:
            # Send the prompt to Hugging Face Inference API
            api_response = query_huggingface(payload)
            response_text = api_response[0].get("generated_text", "Sorry, no response generated.")

            # Extract the response after the [/INST] marker
            response_start = response_text.find("[/INST]")
            if response_start != -1:
                response_after_inst = response_text[response_start + len("[/INST]") :].strip()
            else:
                response_after_inst = "Sorry, no valid answer generated."

            # Return the response as JSON
            return JsonResponse({"response": response_after_inst})

        except Exception as e:
            return JsonResponse({"response": f"Error: {str(e)}"})

    return render(request, "templates/chatbot.html")
