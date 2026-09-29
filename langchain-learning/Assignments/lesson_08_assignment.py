# Part 1

# Load the environment
from dotenv import load_dotenv
load_dotenv()

# Read the model name
import os
MODEL_NAME = os.environ["GEMINI_MODEL"]
API_KEY = os.environ["GOOGLE_GENERATIVE_AI_API_KEY"]

# Create document object
from langchain_core.documents import Document

documents= [
    Document(
    page_content="Data leakage occurs when information from outside the training dataset is inadvertently used to train the model, leading to unrealistically high performance (over fitting) that won’t hold in real-world scenarios",
    metadata ={"source": "Hands on machine learning","topic": "Data leakage" } 
    ),

    Document(
        page_content="LangChain is an open-source software framework that makes it easy to build applications using large language models (LLMs).",
        metadata ={"source": "Wikipedia","topic": "LangChain"} 
    ),
    Document(
        page_content="Retrieval is the process of finding information relevant to a given query.",
        metadata ={"source": "Personal notes","topic": "Retriever"} 
    ),
    Document(
        page_content="Retrieval provides the retrieval interface and uses the underlying retrieval mechanism to return relevant `Document` objects.",
        metadata ={"source": "Personal notes","topic": "Retriever"} 
    ),
    
    Document(
        page_content="Concurrency is the ability of a system to make progress on multiple tasks during overlapping periods of time, by switching between tasks when appropriate. example if A waits for the database while B runs, A and B are progressing concurrently.",
        metadata ={"source": "Personal notes","topic": "Async"} 
    ),
]

# print(documents[0].page_content, "\n",documents[0].metadata )

# Part 2

# Create retriever
from langchain_core.retrievers import BaseRetriever

class TopicRetriever(BaseRetriever):
    stored_documents: list[Document]

    def _get_relevant_documents(self, query:str)-> list[Document]: 
        relevant_documents = [
            doc
            for doc in self.stored_documents
            if doc.metadata["topic"] == query
        ]
        return relevant_documents

retriever = TopicRetriever(stored_documents = documents)

results = retriever.invoke("Retriever")

for doc in results:
    print(doc)

# Part 3

# Create the embedding model
from langchain_google_genai import GoogleGenerativeAIEmbeddings

embeddings = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001",
    google_api_key=API_KEY,
)

# Create the vector store
from langchain_core.vectorstores import InMemoryVectorStore

vector_store = InMemoryVectorStore(embedding=embeddings)

# Add documents to the vector store
vector_store.add_documents(documents)

# Create the retriever
retriever = vector_store.as_retriever(
    search_kwargs={"k":2}
)


query = "What is LangChain"

results = retriever.invoke(query)

print("\n",query,"\n")

for doc in results:
    print(doc.page_content, "\n",doc.metadata )

# part 4

# Similarity retriever
similarity_retriever = vector_store.as_retriever(
    search_type = "similarity",
    search_kwargs = {"k":2}
)

# MMR retriever
mmr_retriever = vector_store.as_retriever(
    search_type = "mmr",
    search_kwargs = {"k":2}
)

similarity_results = similarity_retriever.invoke(query)
mmr_results = mmr_retriever.invoke(query)

for doc in similarity_results:
    print(doc.page_content, "\n",doc.metadata )

print("\n",query,"\n")

for doc in mmr_results:
    print(doc.page_content, "\n",doc.metadata )

# Part 5
metadata_retriever = vector_store.as_retriever(
    search_type = "mmr",
    search_kwargs = {
        "k":2,
        "filter": lambda doc: doc.metadata.get("topic") == "Retriever"
    }
)

metadata_results = metadata_retriever.invoke("What is a retriever?")

print("\n")

for doc in metadata_results:
    print(doc.page_content, "\n",doc.metadata, "\n" )

# part 6

"""
Questions:
1. What is the input/output contract of a Retriever?

2. Why does a Retriever return Document objects instead of just strings?

3. Why is a Retriever a Runnable?

4. What is the difference between invoke() and batch() when used with a Retriever?

5. What is the responsibility of the embedding model?

6. What is the responsibility of the vector store?

7. What is the responsibility of the Retriever?

8. What does k control?

9. What problem does MMR try to solve?

10. What is the difference between metadata filtering
    and semantic similarity?

11. Does a Retriever have to use embeddings?

12. What is the difference between retrieval and RAG?

Answers:
1- Input in a retriever is a query of strings(can be list of strings), the output is list of documents where document have metadata and page content which is the text content

2- Because document is not just a string, it both a string and metadata of the text,  the metadata can include very useful info like the source of the document and the topic of the content. This is very useful in debugging the retriever and when add the retriever to RAG system it enhance the generated answer.
correction: it doesn't necessarily enhance the generated answer by itself, but it gives the RAG system additional information that can be used for things like source attribution, filtering, citations and, context.

3- First to be consistent and usable with the rest of LangChain components, second so it can use the methods of the runnables such as invoke and batch which help in having variety of choices of the input shape.

4- invoke have one input query and one output (which can be a list of documents), batch multi input and multi output

5- Embedding model convert the user query and the documents to numerical representation (vectors)

6- Vector store store and search the vectors (documents numerical representations).

7- Retriever is responsible for the interface that defines the search of the user query.
Better mental model: the retriever provides a standard interface for retrieving relevant documents and delegates teh actual retrieval mechanism to its underlying implementation.

8- K controls the number of the retrieved documents.

9- MMR try to solve the redundancy of the retrieved documents that may happened due to overlapping of chunks or similarity of documents content in general.

10- semantic similarity retrieve documents that have the highest vector similarity score with the input query, while metadata filtering only consider documents that have the same metadata with the retriever's metadata filter.
Correction: only documents that satisfying the specified filter condition are eligible for retrieval.

11- No, embedding based search is one of the strategies that can be used with the retriever but not the only one.

12- Retriever is responsible for return relevant documents based on input query, while RAG is responsible for generate answers based using the retrieved documents based on the input query.

"""

