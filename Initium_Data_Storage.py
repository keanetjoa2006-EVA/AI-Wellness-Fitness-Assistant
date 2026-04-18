import pandas as pd
import chromadb

client = chromadb.PersistentClient(path="./gym_db")
df = pd.read_csv("gym_data.csv")
collection = client.get_or_create_collection(name="gym_knowledge")

def get_gym_data():
    return collection

for index, row in df.iterrows():
    combined = f"Question: {row['question']}\n Answer: {row['answer']}"
    collection.add(
        documents=[combined],
        ids=[f"qa_{index}"],
    )

print(f"{len(df)} records stored in ChromaDB successfully!")