
from langchain_milvus import Milvus

from index_games import local_embeddings
from create_vectordb import DB_PATH


vector_store = Milvus(
    embedding_function=local_embeddings,
    connection_args={ "uri": DB_PATH, },
    index_params={"index_type": "FLAT", "metric_type": "L2"},
    collection_name="games",
)
retriever = vector_store.as_retriever(search_type="mmr", search_kwargs={ "k": 5 })


if __name__ == "__main__":
    
    for doc in retriever.invoke(
        "Chess except every piece is the same, is only on the dark squares, and can only move diagonally."
    ):
        print(doc.metadata["name"])
