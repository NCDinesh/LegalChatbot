from PyPDF2 import PdfReader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
from langchain_community.vectorstores import Chroma
from langchain.schema.runnable import RunnablePassthrough
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.llms import CTransformers
import joblib
import os

# Configuration for LLM
config = {
    'max_new_tokens': 500,  # Adjusted to potentially reduce response time
    'temperature': 0.7,     # Adjusted for more coherent responses
    'context_length' : 2048
}

# Initialize the model
llm = CTransformers(
    model='models/llama-2-7b-chat.ggmlv3.q2_K.bin',
    model_type='llama',
    config=config
)

# Define the prompt template
template = """
<s>[INST] <<SYS>>
You are a helpful AI assistant who has to act as advocate.
Answer based on the context provided. If you cannot find the correct answer, say I don't know. Be concise and just include the response.
<</SYS>>
{context}
Question: {question}
Helpful Answer: [/INST]
"""
prompt = PromptTemplate.from_template(template)

# Load or retrieve preprocessed texts and embeddings
texts_file = "./data/cases_texts.joblib"
embeddings_file = "./data/cases_embeddings.joblib"

if os.path.exists(texts_file) and os.path.exists(embeddings_file):
    # Load preprocessed texts and embeddings
    texts = joblib.load(texts_file)
    embeddings = joblib.load(embeddings_file)
else:
    # Load PDF and extract text
    reader = PdfReader('data/fine_tune_data.pdf')
    raw_text = ''
    for i, page in enumerate(reader.pages):
        text = page.extract_text()
        if text:
            raw_text += text

    # Split text
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=200,
        chunk_overlap=20,
        length_function=len,
    )
    texts = text_splitter.split_text(raw_text)

    # Compute embeddings
    embeddings = HuggingFaceEmbeddings()
    vectorstore = Chroma.from_texts(texts, embeddings)

    # Store texts and embeddings
    joblib.dump(texts, texts_file)
    joblib.dump(embeddings, embeddings_file)

# Initialize vectorstore with loaded embeddings
vectorstore = Chroma.from_texts(texts, embeddings)
retriever = vectorstore.as_retriever()

# Define the query
query = "My teacher intentionally touches me time and again , what should i do?"

# Retrieve the relevant documents
retrieved_docs = retriever.get_relevant_documents(query)

# Construct the context from the retrieved documents
context = "\n".join([doc.page_content for doc in retrieved_docs])

# Prepare the input for the chain
input_data = {"context": context, "question": query}

# Define the chain
chain = (
    {"context": RunnablePassthrough(), "question": RunnablePassthrough()}
    | prompt 
    | llm 
)

# Invoke the chain and get the response
response = chain.invoke(input_data)
print(response)
