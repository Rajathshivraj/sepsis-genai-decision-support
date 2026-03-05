from src.rag.cosine_retrieval import retrieve_similar_cases_cosine

query = "Patient HR 120 Temp 39 MAP 60 Lactate 4.0"

cases = retrieve_similar_cases_cosine(query, k=5)

print("\nRetrieved Cases\n")

for c in cases:
    print(c)

