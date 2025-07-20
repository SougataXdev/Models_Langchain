from langchain_huggingface import HuggingFaceEmbeddings

embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")


text = "kolkata is the capital of wb"


# res = embedding.embed_query(text=text)

# print(str(res))


# how to embedd documents


document = [
    "i am sougata",
    "he is a doctor"
]


res = embedding.embed_documents(document)


print(str(res))