import bs4
from langchain import hub
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.llms.ollama import Ollama
from langchain_community.vectorstores import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter


# Load
loader = WebBaseLoader(
    web_paths=("https://lilianweng.github.io/posts/2023-06-23-agent/",),
    bs_kwargs=dict(
        parse_only=bs4.SoupStrainer(
            class_=("post-content", "post-title", "post-header")
        )
    ),
)
docs = loader.load()

# transform
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
splits = text_splitter.split_documents(docs)

# embedding
ollama_embeddings = OllamaEmbeddings(base_url="http://localhost:11434", model="nomic-embed-text")

# store
# create collection
# vectorstore = Chroma.from_documents(documents=splits, embedding=ollama_embeddings, persist_directory="./chroma_RAG")
# or get collection
vectorstore = Chroma(persist_directory="./chroma_RAG", embedding_function=ollama_embeddings)

# Retrieve and generate using the relevant snippets of the blog.
retriever = vectorstore.as_retriever()

prompt = hub.pull("rlm/rag-prompt")

llm = Ollama(base_url="http://localhost:11434", model="qwen:7b", temperature=0.5)


def format_docs(docs):
    res = "\n————————————————————————————————————————————————\n".join(doc.page_content for doc in docs)
    print(f'召回的内容是:\n\n{res}')
    print("———————————————以上来自召回———————————————————————")
    return res


rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

res = rag_chain.invoke("What is Task Decomposition?")

print(res)

"""
召回的内容是:

Fig. 1. Overview of a LLM-powered autonomous agent system.
Component One: Planning#
A complicated task usually involves many steps. An agent needs to know what they are and plan ahead.
Task Decomposition#
Chain of thought (CoT; Wei et al. 2022) has become a standard prompting technique for enhancing model performance on complex tasks. The model is instructed to “think step by step” to utilize more test-time computation to decompose hard tasks into smaller and simpler steps. CoT transforms big tasks into multiple manageable tasks and shed lights into an interpretation of the model’s thinking process.
————————————————————————————————————————————————
Fig. 11. Illustration of how HuggingGPT works. (Image source: Shen et al. 2023)
The system comprises of 4 stages:
(1) Task planning: LLM works as the brain and parses the user requests into multiple tasks. There are four attributes associated with each task: task type, ID, dependencies, and arguments. They use few-shot examples to guide LLM to do task parsing and planning.
Instruction:
————————————————————————————————————————————————
(3) Task execution: Expert models execute on the specific tasks and log results.
Instruction:

With the input and the inference results, the AI assistant needs to describe the process and results. The previous stages can be formed as - User Input: {{ User Input }}, Task Planning: {{ Tasks }}, Model Selection: {{ Model Assignment }}, Task Execution: {{ Predictions }}. You must first answer the user's request in a straightforward manner. Then describe the task process and show your analysis and model inference results to the user in the first person. If inference results contain a file path, must tell the user the complete file path.
————————————————————————————————————————————————
The AI assistant can parse user input to several tasks: [{"task": task, "id", task_id, "dep": dependency_task_ids, "args": {"text": text, "image": URL, "audio": URL, "video": URL}}]. The "dep" field denotes the id of the previous task which generates a new resource that the current task relies on. A special tag "-task_id" refers to the generated text image, audio and video in the dependency task with id as task_id. The task MUST be selected from the following options: {{ Available Task List }}. There is a logical relationship between tasks, please note their order. If the user input can't be parsed, you need to reply empty JSON. Here are several cases for your reference: {{ Demonstrations }}. The chat history is recorded as {{ Chat History }}. From this chat history, you can find the path of the user-mentioned resources for your task planning.
———————————————以上来自召回———————————————————————
As an AI assistant, I understand that you need help with Task Decomposition.

Task Planning involves parsing user requests into multiple tasks. Each task has specific attributes such as task type, ID, dependencies, and arguments.

For example, if a user asks to generate a report on sales data, the tasks could be:
1. Parse the request for report generation.
2. Extract sales data from an available source.
3. Analyze the extracted data to create meaningful insights.
4. Generate a report in a specified format (e.g., PDF).

The dependencies would indicate which task generates a resource needed by another task.

In conclusion, Task Decomposition is the process of breaking down a complex task into smaller, manageable steps. This enables AI assistants like myself to plan and execute tasks efficiently.

"""