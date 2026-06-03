import os
from dotenv import load_dotenv
from src.vectorstore import FaissVectorStore
from langchain_groq import ChatGroq

load_dotenv()


class RAGSearch:
    def __init__(
        self,
        persist_dir="faiss_store",
        embedding_model="all-MiniLM-L6-v2",
        llm_model="llama-3.3-70b-versatile"
    ):

        self.vectorstore = FaissVectorStore(
            persist_dir=persist_dir,
            embedding_model=embedding_model
        )

        faiss_path = os.path.join(persist_dir, "faiss_index")
        meta_path = os.path.join(persist_dir, "metadata.pkl")

        if not (os.path.exists(faiss_path) and os.path.exists(meta_path)):
            from data_loader import load_all_documents

            docs = load_all_documents("data")
            self.vectorstore.build_from_documents(docs)
        else:
            self.vectorstore.load()

        print("Metadata entries:", len(self.vectorstore.metadata))
        if self.vectorstore.metadata:
            print("First metadata:", self.vectorstore.metadata[0])

        groq_api_key = ("")

        self.llm = ChatGroq(
            api_key=groq_api_key,
            model_name=llm_model,
            temperature=0.1,
            max_tokens=1024,
        )

        print(f"[INFO] Groq LLM initialized with model: {llm_model}")
        
    def search_and_summarize(self, query: str, top_k: int = 5):

        results = self.vectorstore.query(query, top_k=top_k)

        print("\nRetrieved results:")
        for r in results:
            print(r)

        texts = [
            r["metadata"].get("text", "")
            for r in results
            if r.get("metadata")
        ]

        context = "\n\n".join(texts)

        if not context:
            return "No relevant documents found."

        prompt = f"""
            Use the following context to answer the question concisely.

            Context:
            {context}

            Question:
            {query}

            Answer:
            """

        response = self.llm.invoke(prompt)

        return response.content

#class RAGSearch:
    # def __init__(self, persist_dir: str = "faiss_store", embedding_model: str = "text-embedding-3-small", llm_model: str = "llama-3.3-70b-versatile"):
    #     faiss_path = os.path.join(persist_dir, "faiss_index.faiss")
    #     meta_path = os.path.join(persist_dir, "metadata.pkl")
    #     if not (os.path.exists(faiss_path) and os.path.exists(meta_path)):
    #         from data_loader import load_all_documents
    #         docs = load_all_documents("data")
    #         self.vector_store.build_from_documents(docs)
    #     else:
    #         self.vectorstore.load()
    #     groq_api_key = ""
    #     self.llm = ChatGroq(api_key=groq_api_key, model_name = "llama-3.3-70b-versatile", temperature=0.1, max_tokens=1024)
    #     print(f"[INFO] Groq LLM initialized with model: {llm_model}")
        
    # def search_and_summarize(self, query: str, top_k: int = 5) -> str:
    #     results = self.vectorstore.query(query, top_k=top_k)
    #     texts = [r["metadata"].get("texts,"") for r in results if "metadata" in r and "texts" in r["metadata"]]
    #     context = "\n\n".join(texts)
    #     if not context:
    #         return "No relevant documents found."
    #     prompt = """Use the following context to answer the question concisely.
    #     Context:
    #     {context}
    
    #     Question: {query}
    #     Answer:"""
        
    #     response = self.llm.invoke([prompt])
    #     return response.content    
        
        
                                   
        
        
        
        
        
        
        
        
        
        
      