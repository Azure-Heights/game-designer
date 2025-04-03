
import subprocess

from dotenv import load_dotenv
load_dotenv()

from langchain_core.runnables import RunnablePassthrough
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.schema import StrOutputParser

from langchain_milvus import Milvus
from langchain_openai import ChatOpenAI

from index_games import local_embeddings
from create_vectordb import DB_PATH


COMPILER_PATH = "scripts/AutoCompile.jar"
OUT_PATH = "output/generated.lud"


def example_formatter(examples):
    return "\n\n".join([
        str({ **ex.metadata, "rules": ex.page_content })
        for ex in examples
    ])

def compile(ludeme):
    filename = OUT_PATH
    with open(filename, "w", encoding="utf-8") as f:
        f.write(ludeme)
    result = subprocess.run(["java", "-jar", COMPILER_PATH, filename], capture_output=True, text=True)
    return ("Success\n" == result.stdout) or result.stderr


gen_prompt = ChatPromptTemplate.from_messages([
    ("system",
     """You are tasked with coming up with a code file for a particular game idea. The following
     message contains examples of fully complete, working examples written in a language called
     Ludii, a declarative language designed specifically for writing games.

     Select one of the examples to use as a base, and modify it to fit the user's idea as well as
     possible. In order to avoid writing incorrect code, only use snippets attested to by the
     examples. Use only functions and classes that you see in the examples. Do not make up functions
     that you don't see. If you see a function defined and used in an example, make sure you define
     it too. Duplicate all helper functions. Duplicate all additional definitions. Don't vary too
     much from the example you choose.

     Respond only with the Ludii code, with no code block, so that it can be output straight into a
     file on disk and compiled.
     """),
    # MessagesPlaceholder(variable_name=examples),
    ("system", "Examples:\n\n{examples}"),
    ("human", "{query}"),
])


llm = ChatOpenAI(model="gpt-4o")

vector_store = Milvus(
    embedding_function=local_embeddings,
    connection_args={ "uri": DB_PATH, },
    index_params={"index_type": "FLAT", "metric_type": "L2"},
    collection_name="games",
)
retriever = vector_store.as_retriever(search_type="mmr", search_kwargs={ "k": 5 })

chain = (
    { "query": RunnablePassthrough(), "examples": retriever | example_formatter }
    | RunnablePassthrough.assign(ludeme=gen_prompt | llm | StrOutputParser())
    | RunnablePassthrough.assign(success=lambda s: compile(s["ludeme"]))
)


if __name__ == "__main__":

    result = chain.invoke("Create a chess variant where the back row is full of kings and there are two rows of pawns. Do not make mention of any other pieces.")

    print(f"{result["ludeme"]}\n=== {'SUCCESS' if result['success'] else 'FAILURE'} ===")

