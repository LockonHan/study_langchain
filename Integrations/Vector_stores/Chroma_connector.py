import chromadb
from chromadb.config import Settings


class ChromaDBConnector:
    def __init__(self, collection_name, embedding):
        self.persistent_client = chromadb.PersistentClient(path="./chroma_db")

        # 创建一个 collection
        self.collection = self.persistent_client.get_or_create_collection(name=collection_name)
        self.embedding = embedding

    def add_documents(self, docs):
        """向 collection 中添加文档与向量"""
        self.collection.add(
            embeddings=self.get_embeddings(docs),  # 每个文档的向量
            documents=docs,  # 文档的原文
            ids=[f"id{i}" for i in range(len(docs))]  # 每个文档的 id
        )

    def search(self, query, top_n=10):
        """检索向量数据库"""
        results = self.collection.query(
            query_embeddings=self.get_embeddings([query]),
            n_results=top_n
        )
        return results


    def get_embeddings(self, docs):
        """封装 Ollama 的 Embedding 模型接口"""
        return self.embedding.embed_documents(docs)
