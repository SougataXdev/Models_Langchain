from langchain_huggingface import HuggingFaceEndpointEmbeddings
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

embeddings = HuggingFaceEndpointEmbeddings(
    model="sentence-transformers/all-MiniLM-L6-v2",
    huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_API_TOKEN")
)

text = "Kolkata is the capital of WB"
embedding = embeddings.embed_query(text)
print(embedding[:5])