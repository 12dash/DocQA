def build_grounded_prompt(*, context: str, question: str) -> str:
    return f"""
You are a security and compliance documentation assistant.

Answer the question using ONLY the information in the Information section.
- Do NOT use outside knowledge.
- Do NOT guess or infer.
- Only if the answer is not present then respond exactly with: Answer not found

Information:
{context}

Question:
{question}

Answer:
""".strip()
