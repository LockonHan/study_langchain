from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import Chroma
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

# embedding
ollama_embeddings = OllamaEmbeddings(base_url="http://localhost:11434", model="nomic-embed-text")

# store
# 创建一个chroma的数据库实例，并且配置一个地址
vectorstore = Chroma.from_documents(documents=splits, embedding=ollama_embeddings, persist_directory="./chroma_db")

# Retrieve and generate using the relevant snippets of the blog.
retriever = vectorstore.as_retriever()
prompt = hub.pull("rlm/rag-prompt")
llm = Ollama(base_url="http://localhost:11434", model="qwen:4b", temperature=0.5)


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
)

for chunk in rag_chain.stream("What is Task Decomposition?"):
    print(chunk, end="", flush=True)
