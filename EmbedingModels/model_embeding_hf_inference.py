from huggingface_hub import InferenceClient
from dotenv import load_dotenv
import os

load_dotenv()

token = os.getenv("HUGGINGFACEHUB_API_TOKEN")

client = InferenceClient(token=token)

model_id = "sentence-transformers/all-MiniLM-L6-v2"
text = "I am Sougata"

res = client.feature_extraction(text, model=model_id)

print(res[:5]) 
