import chromadb as chroma


# from langchain_community.vectorstores import Chroma as chroma


class ChromaDBConnector:
    def __init__(self, collection_name, embedding):
        self.persistent_client = chroma.PersistentClient(path="./chroma_RAG_test")
        self.collection = self.persistent_client.get_or_create_collection(collection_name)
        self.embedding = embedding

    def add_documents(self, docs):
        """向 collection 中添加文档与向量"""
        print("开始添加文档向量到向量数据库...")
        self.collection.add(
            embeddings=self.get_embeddings(str(doc.page_content) for doc in docs),  # 每个文档的向量
            documents=[str(doc.page_content) for doc in docs],  # 文档的原文
            ids=[print(f"id{docs[i].metadata.get('start_index')}") or f"id{docs[i].metadata.get('start_index')}" for i in range(len(docs))]  # 每个文档的 id
        )
        print("添加文档向量完成")

    def search(self, query):
        # print(f'query: {query}')
        """检索向量数据库"""
        results = self.collection.query(
            query_embeddings=self.get_embeddings([query]),
            n_results=4
        )
        # for para in results['documents'][0]:
        #     print(para + "\n")
        return results

    def get_embeddings(self, docs):
        """封装 Ollama 的 Embedding 模型接口"""
        return self.embedding.embed_documents(docs)
