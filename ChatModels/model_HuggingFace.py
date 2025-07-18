from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="moonshotai/Kimi-K2-Instruct",
    task="text-generation"
)

model = ChatHuggingFace(llm=llm)


res = model.invoke("what is capital of america")

print(res.content)