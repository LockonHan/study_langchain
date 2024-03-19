from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import Chroma
from Integrations.Vector_stores.Chroma_connector import ChromaDBConnector
from langchain_community.embeddings import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_core.runnables import RunnablePassthrough
from langchain_community.llms import Ollama
from langchain_core.output_parsers import StrOutputParser

import bs4
from langchain import hub


# load
loader = WebBaseLoader(
    web_paths=("https://lilianweng.github.io/posts/2023-06-23-agent/",),
    bs_kwargs=dict(
        parse_only=bs4.SoupStrainer(
            class_=("post-content", "post-title", "post-header")
        )
    )

)
docs = loader.load()

# transform
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
splits = text_splitter.split_documents(docs)
splits_json = [spilt.to_json() for spilt in splits]

# embedding
ollama_embeddings = OllamaEmbeddings(base_url="http://localhost:11434", model="nomic-embed-text")

# store
# create collection
# vectorstore = Chroma.from_documents(documents=splits, embedding=ollama_embeddings, persist_directory="./chroma_db")
# or get collection
# vectorstore = Chroma(persist_directory="./chroma_db", embedding_function=ollama_embeddings)

vectorstore = ChromaDBConnector(collection_name="test", embedding=ollama_embeddings)
# vectorstore.add_documents(splits_json)

# Retrieve and generate using the relevant snippets of the blog.
retriever = vectorstore.search

prompt = hub.pull("rlm/rag-prompt")
llm = Ollama(base_url="http://localhost:11434", model="qwen:7b", temperature=0.5)


def format_docs(docs):
    return "\n\n".join(doc for doc in docs)


res = vectorstore.search("What is Task Decomposition?")
print(format_docs(res))

rag_chain = (
        {"context": vectorstore.search(RunnablePassthrough()) | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
)

for chunk in rag_chain.stream("What is Task Decomposition?"):
    print(chunk, end="", flush=True)

# """
# qwen:4b:
# I'm sorry, but I don't know what Task Decomposition is. Can you please provide more context or information about Task Decomposition?
# qwen:7b:
# Task Decomposition involves breaking down a complex task into smaller, manageable steps. This process helps an agent or system plan ahead and execute tasks more efficiently.
#
# For example, if a user asks to book a flight, this complex task can be decomposed as follows:
#
# 1. Gather user input: Collect information about the flight such as departure city, destination city, travel dates, and preferred airline.
#
# 2. Search for available flights: Use the gathered information to search for flights that match the user's criteria.
#
# 3. Filter results: Apply filters to the search results based on factors like price, seat availability, layovers, and time of day.
#
# 4. Present flight options: Display a list of flight options to the user, including details such as flight number, departure/arrival times, and prices.
#
# 5. Assist with booking: If the user selects a flight and proceeds with the booking process, assist them by guiding through payment steps and providing any additional information or support needed.
# """
