
import json

import numpy as np

from pymilvus import (
    Collection,
    CollectionSchema,
    DataType,
    FieldSchema,
    connections,
)
from pymilvus.exceptions import MilvusException

from langchain_core.documents import Document
from langchain_milvus import Milvus
from langchain_milvus.utils.sparse import BM25SparseEmbedding

from index_games import INDEX_OUTPUT_PATH, local_embeddings


DB_PATH = "data/milvus.db"
vector_store = Milvus(
    embedding_function=local_embeddings,
    connection_args={ "uri": DB_PATH, },
    index_params={"index_type": "FLAT", "metric_type": "L2"},
)

# +-- manual attempt
# connections.connect(uri=DB_PATH)


# schema = CollectionSchema(fields=[
#         FieldSchema(
#             name="game_id",
#             dtype=DataType.VARCHAR,
#             is_primary=True,
#             auto_id=True,
#             max_length=100,
#         ),
#         FieldSchema(name="name",   dtype=DataType.VARCHAR, max_length=200),
#         FieldSchema(name="rules",  dtype=DataType.VARCHAR, max_length=500),
#         FieldSchema(name="text",   dtype=DataType.VARCHAR, max_length=10_000),
#         FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=768),
#     ],
#     enable_dynamic_fields=False,
# )
# collection = Collection(name="games", schema=schema)
# --+


if __name__ == "__main__":

    with open(INDEX_OUTPUT_PATH, "r") as fin:
        data = json.load(fin)

    # +--
    # entities = [{
    #     "name":   name,
    #     "rules":  game["rules"],
    #     "text":   game["ludeme"],
    #     "vector": game["embedding"][0],
    # } for name, game in data.items() ]

    # collection.insert(entities)
    # --+

    documents = [
        Document(
            page_content=game["rules"],
            metadata={
                "name":   name,
                "ludeme": game["ludeme"],
            },
        ) for name, game in data.items()
    ]

    games_collection = Milvus.from_documents(
        documents,
        local_embeddings,
        collection_name="games",
        connection_args={ "uri": DB_PATH, },
        index_params={"index_type": "FLAT", "metric_type": "L2"},
    )

