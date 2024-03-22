from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import Chroma
from Integrations.Vector_stores.Chroma_connector import ChromaDBConnector
from langchain_community.embeddings import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_core.runnables import RunnablePassthrough, RunnableLambda
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
#  set add_start_index=True so that the character index at which each split Document starts
#  within the initial Document is preserved as metadata attribute “start_index”.
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200, add_start_index=True)
splits = text_splitter.split_documents(docs)

# embedding
ollama_embeddings = OllamaEmbeddings(base_url="http://localhost:11434", model="nomic-embed-text")

# store
# create collection
# vectorstore = Chroma.from_documents(documents=splits, embedding=ollama_embeddings, persist_directory="./chroma_db")
# or get collection
# vectorstore = Chroma(persist_directory="./chroma_db", embedding_function=ollama_embeddings)

vectorstore = ChromaDBConnector(collection_name="test", embedding=ollama_embeddings)
# vectorstore.add_documents(splits)

# Retrieve and generate using the relevant snippets of the blog.
retriever = vectorstore.search

prompt = hub.pull("rlm/rag-prompt")
llm = Ollama(base_url="http://localhost:11434", model="qwen:7b", temperature=0.5)


def format_docs(docs):
    res = "\n————————————————————————————————————————————————\n".join(doc for doc in docs['documents'][0])
    print(f'召回的内容是:\n\n{res}')
    print("———————————————以上来自召回———————————————————————")
    return res

rag_chain = (
        {"context": RunnablePassthrough() | RunnableLambda(retriever) | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
)

for chunk in rag_chain.stream("What is Task Decomposition?"):
    print(chunk, end="", flush=True)

"""
召回的内容是:

Fig. 1. Overview of a LLM-powered autonomous agent system.
Component One: Planning#
A complicated task usually involves many steps. An agent needs to know what they are and plan ahead.
Task Decomposition#
Chain of thought (CoT; Wei et al. 2022) has become a standard prompting technique for enhancing model performance on complex tasks. The model is instructed to “think step by step” to utilize more test-time computation to decompose hard tasks into smaller and simpler steps. CoT transforms big tasks into multiple manageable tasks and shed lights into an interpretation of the model’s thinking process.
————————————————————————————————————————————————
Tree of Thoughts (Yao et al. 2023) extends CoT by exploring multiple reasoning possibilities at each step. It first decomposes the problem into multiple thought steps and generates multiple thoughts per step, creating a tree structure. The search process can be BFS (breadth-first search) or DFS (depth-first search) with each state evaluated by a classifier (via a prompt) or majority vote.
Task decomposition can be done (1) by LLM with simple prompting like "Steps for XYZ.\n1.", "What are the subgoals for achieving XYZ?", (2) by using task-specific instructions; e.g. "Write a story outline." for writing a novel, or (3) with human inputs.
————————————————————————————————————————————————
(3) Task execution: Expert models execute on the specific tasks and log results.
Instruction:

With the input and the inference results, the AI assistant needs to describe the process and results. The previous stages can be formed as - User Input: {{ User Input }}, Task Planning: {{ Tasks }}, Model Selection: {{ Model Assignment }}, Task Execution: {{ Predictions }}. You must first answer the user's request in a straightforward manner. Then describe the task process and show your analysis and model inference results to the user in the first person. If inference results contain a file path, must tell the user the complete file path.
————————————————————————————————————————————————
(2) Model selection: LLM distributes the tasks to expert models, where the request is framed as a multiple-choice question. LLM is presented with a list of models to choose from. Due to the limited context length, task type based filtration is needed.
Instruction:

Given the user request and the call command, the AI assistant helps the user to select a suitable model from a list of models to process the user request. The AI assistant merely outputs the model id of the most appropriate model. The output must be in a strict JSON format: "id": "id", "reason": "your detail reason for the choice". We have a list of models for you to choose from {{ Candidate Models }}. Please select one model from the list.

(3) Task execution: Expert models execute on the specific tasks and log results.
Instruction:
———————————————以上来自召回———————————————————————
To complete the task, I need to follow the steps mentioned in the context provided.

1. **Planning** - Since the question is about Task Decomposition, I understand that this step involves breaking down a complex task into smaller manageable steps.

2. **Task Decomposition** - This can be achieved using language models like LLM, as demonstrated by the examples given: (1) prompting LLM directly with simple instructions like "Steps for XYZ.", or (2) using task-specific instructions for specific tasks like writing a novel or creating a story outline.

3. **Model Selection** - In this step, I would present a list of models to the user and choose the most appropriate one based on the limited context length and task type.

4. **Task Execution** - Once the model is selected, it would execute the specific task and log the results.

In conclusion, for Task Decomposition, I would use language models like LLM, present them with task instructions, select the best model, and have it execute the task and log the results.
"""
