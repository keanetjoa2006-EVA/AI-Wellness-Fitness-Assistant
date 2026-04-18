import chromadb
import requests
import json

client = chromadb.PersistentClient(path="./gym_db")
collection = client.get_or_create_collection(name="gym_knowledge")

def extract_answer(document_text):
    for line in document_text.splitlines():
        if line.strip().lower().startswith("answer:"):
            return line.split(":", 1)[1].strip()
    return document_text.strip()

def get_exact_answer(user_question):
    data = collection.get()
    docs = data.get("documents", [])
    if docs and isinstance(docs[0], list):
        docs = docs[0]

    for doc in docs:
        if not isinstance(doc, str):
            continue
        if "Question:" not in doc:
            continue
        parts = doc.split("\n Answer:", 1)
        if len(parts) != 2:
            continue
        q_text = parts[0].replace("Question:", "", 1).strip().strip('"')
        if q_text.lower() == user_question.strip().lower():
            return parts[1].strip()
    return None

def search_knowledge_base(user_question, num_results=3, relevance_threshold=1.5):
    results = collection.query(
        query_texts=[user_question],
        n_results=num_results
    )
    if not results["documents"]:
        return []
    
    docs = results["documents"][0]
    distances = results.get("distances", [[]])[0] if results.get("distances") else []
    
    relevant = []
    for doc, distance in zip(docs, distances):
        if distance < relevance_threshold:
            relevant.append(doc)
    
    return relevant

def needs_more_context(user_question):
    """Check if a question is too ambiguous or lacks specifics."""
    prompt = f"""Analyze this question briefly. Does it lack sufficient context or specifics to provide a high-quality answer?
    
Question: {user_question}

Respond with only 'YES' if it needs more context, or 'NO' if it has enough info."""
    
    response = requests.post(
        "http://localhost:11434/api/generate",
        json={
            "model": "llama3",
            "prompt": prompt,
            "stream": False
        }
    )
    result = response.json().get("response", "").strip().upper()
    return "YES" in result


def ask_for_context(user_question):
    """Ask clarifying questions to get more context."""
    prompt = f"""You are a helpful fitness assistant. The user asked a vague or incomplete question.
Ask 1-2 brief, specific clarifying questions to get the context needed for a better answer.
Keep questions direct and actionable.

User's question: {user_question}

Clarifying questions:"""
    
    response = requests.post(
        "http://localhost:11434/api/generate",
        json={
            "model": "llama3",
            "prompt": prompt,
            "stream": False
        }
    )
    return response.json().get("response", "")


def form_answer_with_context(user_question, context_info):
    """Form a stronger answer using the provided context."""
    exact_answer = get_exact_answer(user_question)
    related_docs = search_knowledge_base(user_question, num_results=3)
    
    if exact_answer:
        answers = [extract_answer(doc) for doc in related_docs if doc]
        gym_context = "\n".join(answers) if answers else ""
        
        prompt = f"""You are a certified personal trainer AI assistant.
The user provided additional context to clarify their question.
Use this context along with the base answer and any related fitness information to provide a comprehensive, tailored response.

Base answer: {exact_answer}

Additional user context: {context_info}

Related fitness info: {gym_context}

Question: {user_question}

Comprehensive answer:"""
    elif related_docs:
        answers = [extract_answer(doc) for doc in related_docs if doc]
        gym_context = "\n".join(answers) if answers else ""
        
        prompt = f"""You are a certified personal trainer AI assistant.
The user provided additional context to clarify their question.
Use this context along with relevant fitness information to provide a comprehensive, tailored response.

Additional user context: {context_info}

Related fitness info: {gym_context}

Question: {user_question}

Answer:"""
    else:
        prompt = f"""You are a certified personal trainer AI assistant.
The user provided additional context to clarify their question.
Use this context and your fitness knowledge to provide a comprehensive, tailored response.

Additional user context: {context_info}

Question: {user_question}

Answer:"""
    
    response = requests.post(
        "http://localhost:11434/api/generate",
        json={
            "model": "llama3",
            "prompt": prompt,
            "stream": False
        }
    )
    return response.json().get("response", "")


def ask_gym_ai(user_question, stream=False):
    exact_answer = get_exact_answer(user_question)
    related_docs = search_knowledge_base(user_question, num_results=3)
    
    if exact_answer:
        answers = [extract_answer(doc) for doc in related_docs if doc]
        context = "\n".join(answers) if answers else ""
        
        prompt = f"""You are a certified personal trainer AI assistant.
You have a base answer that directly matches the user's question.
Refine and enhance this answer to be clearer, more helpful, and more accurate.
Preserve the original meaning and core information.
If related fitness information is provided, incorporate it only if it supports and strengthens the answer.
Keep the enhancement concise and professional.

Base answer: {exact_answer}

Related information (optional): {context}

Question: {user_question}

Refined answer:"""
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": "llama3",
                "prompt": prompt,
                "stream": stream
            },
            stream=stream
        )

        if stream:
            return response
        return response.json().get("response", "")

    if related_docs:
        answers = [extract_answer(doc) for doc in related_docs if doc]
        context = "\n".join(answers) if answers else ""
        
        prompt = f"""You are a certified personal trainer AI assistant.
Use the following relevant information to answer the question accurately.
Only reference the information provided if it directly answers the question.
Do not mention or reference any documents, labels, or source format.

Relevant information:
{context}

Question: {user_question}

Answer:"""
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": "llama3",
                "prompt": prompt,
                "stream": stream
            },
            stream=stream
        )

        if stream:
            return response
        return response.json().get("response", "")

    prompt = f"""You are a certified personal trainer AI assistant.
You only answer fitness, gym, nutrition, and health-related questions.
Be specific, practical, and safe in your advice.
Answer the question using your best fitness knowledge.

Question: {user_question}

Answer:"""
    response = requests.post(
        "http://localhost:11434/api/generate",
        json={
            "model": "llama3",
            "prompt": prompt,
            "stream": stream
        },
        stream=stream
    )

    if stream:
        return response
    return response.json().get("response", "")


if __name__ == "__main__":
    while True:
        question = input("\nAsk your gym AI: ")
        if question.lower() == "quit":
            break
        answer_response = ask_gym_ai(question, stream=False)
        if isinstance(answer_response, str):
            print(f"\nAI: {answer_response}")
        else:
            answer = ""
            for line in answer_response.iter_lines():
                if not line:
                    continue
                if isinstance(line, bytes):
                    line = line.decode('utf-8')
                line = line.strip()
                if not line:
                    continue
                if line.startswith("data:"):
                    line = line[5:].strip()
                try:
                    data = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if data.get("done", False):
                    break
                answer += data.get("response", "") or data.get("text", "") or ""
            print(f"\nAI: {answer}")