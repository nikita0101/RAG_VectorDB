from multiprocessing import context
import os
from langchain_community.document_loaders import PyPDFLoader, PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pathlib import Path
import numpy as np
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings
import uuid
from typing import List, Dict, Any, Tuple
from sklearn.metrics.pairwise import cosine_similarity

### Read all the pdfs inside the directory

def process_all_pdfs(pdf_directory):
    all_documents = []
    pdf_dir = Path(pdf_directory)
    
    pdf_files = list(pdf_dir.glob("*.pdf"))
    
    print(f"Found {len(pdf_files)} PDF files to process.")
    
    for pdf_file in pdf_files:
        print(f"\nProcessing {pdf_file.name}")
        try:
            loader = PyPDFLoader(str(pdf_file))
            documents = loader.load()
            
            for doc in documents:
                doc.metadata["source"] = str(pdf_file)
                doc.metadata["file_type"] = 'pdf'
                
            all_documents.extend(documents)
            print(f"Successfully loaded {len(documents)}.")
        except Exception as e:
            print(f"Error loading{e}")
    
    print(f"\nTotal documents loaded: {len(all_documents)}")
    return all_documents

all_pdf_documents = process_all_pdfs("data/pdf")

all_pdf_documents

### Text splitting get into chunks

def split_documents(documents, chunk_size=1000, chunk_overlap=200):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, 
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", " ", ""]
        )
    
    split_docs = text_splitter.split_documents(documents)
    print(f"split: {len(documents)} documents into {len(split_docs)} chunks.")
    
    #Show example of the first chunk
    if split_docs:
        print("\nExample of the first chunk:")
        print(f"Content: {split_docs[0].page_content[:200]}...")  # Print the first 200 characters
        print(f"Metadata: {split_docs[0].metadata}") 
    
    return split_docs


chunks = split_documents(all_pdf_documents)
print(chunks)


class EmbeddingManager:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        self.mode = None
        self._load_model()
    
    def _load_model(self):
        try:
            print(f"Loading embedding model: {self.model_name}")
            self.model = SentenceTransformer(self.model_name)
            print("Model loaded successfully. Embedding dimension: {self.model.get_sentence_embedding_dimension()}")
            
        except Exception as e:
            print(f"Error loading model {self.model_name}: {e}")
            raise
    
    def generate_embeddings(self, texts: List[str]):
        
        print(f"Generating embeddings for {len(texts)} texts...")
        embeddings = self.model.encode(texts, show_progress_bar=True)
        print(f"Generate embeddings with shape: {embeddings.shape}")
        return embeddings

Embedding_manager = EmbeddingManager()
print(Embedding_manager)


class VectorStore:
    def __init__(self, collection_name: str = "pdf_documents", persist_directory: str = "/data/vector_store"):
        self.collection_name = collection_name
        self.persist_directory = persist_directory
        self.client = None
        self.collection = None
        self._initialize_store()
    
    def _initialize_store(self):
        try:
            os.makedirs(self.persist_directory, exist_ok=True)
            self.client = chromadb.PersistentClient(path=self.persist_directory)
            
            self.collection = self.client.get_or_create_collection(
                name=self.collection_name,
                metadata={"description": "PDF document embeddings for RAG"}
            )
            
            print(f"Vector Store initialized. Collection: {self.collection_name}")
            print(f"Existing documents in collection: {self.collection.count()}")
            
        except Exception as e:
            print(f"Error initializing vector store: {e}")
            raise
        
    # def add_documents(self, documents: list[Any], embeddings: np.ndarray):
    
    def add_documents(self, documents: list[Any], embeddings: np.ndarray):
        if len(documents) != len(embeddings):
            raise ValueError("The number of documents must match the number of embeddings.")

        print(f"Adding {len(documents)} documents to the vector store...")

        ids = []
        metadatas = []
        documents_text = []
        embeddings_list = []

        for i, (doc, embedding) in enumerate(zip(documents, embeddings)):
            doc_id = f"doc_{uuid.uuid4().hex[:8]}_{i}"
            ids.append(doc_id)

            meta = dict(doc.metadata)
            meta["doc_index"] = i
            meta["content_length"] = len(doc.page_content)

            metadatas.append(meta)
            documents_text.append(doc.page_content)
            embeddings_list.append(embedding.tolist())

        try:
            self.collection.add(
                ids=ids,
                embeddings=embeddings_list,
                metadatas=metadatas,
                documents=documents_text
            )

            print(f"Successfully added {len(documents)} documents to the vector store.")
            print(f"Total documents in collection after addition: {self.collection.count()}")

        except Exception as e:
            print(f"Error adding documents to vector store: {e}")
            raise

vectorstore= VectorStore()   
vectorstore  

### Convert text to embeddings
texts = [doc.page_content for doc in chunks] 
texts

## generate teh embeddings
embeddings = Embedding_manager.generate_embeddings(texts)
print(f"Embeddings generated with shape: {embeddings.shape}")

## Store in the Vector DB
vectorstore.add_documents(chunks, embeddings)

class RAGRetriver:
    def __init__(self, vector_store: VectorStore, embedding_manager: EmbeddingManager):
        
        self.vector_store = vector_store
        self.embedding_manager = embedding_manager
        
    def retrieve(self, query: str, top_k: int = 5, score_threshold: float = 0.0) -> List[Dict[str, Any]]:
        print(f"Retrieving documents for query: '{query}'")
        print(f"Top K: {top_k}, Score Threshold: {score_threshold}")
        
        #Generate query embedding
                
        query_embedding = self.embedding_manager.generate_embeddings([query])[0]
        print(f"Query embedding generated with shape: {query_embedding.shape}")
        
        try:
            results = self.vector_store.collection.query(
                query_embeddings=[query_embedding.tolist()],
                n_results=top_k,
             )
            
            #Process Results            
                
            retrieved_docs = []
            if results['documents'] and results['documents'][0]:
                documents= results['documents'][0]
                metasdatas = results['metadatas'][0]
                distances = results['distances'][0]
                ids = results['ids'][0]
                
                for i, (doc_id, documents, metadata, distance) in enumerate(zip(ids, documents, metasdatas, distances)):
                    
                    similarity_score = 1 - distance
                    if similarity_score >= score_threshold:
                        retrieved_docs.append({
                            "id": doc_id,
                            'content': documents,
                            "metadata": metadata,
                            "similarity_score": similarity_score,
                            "distance": distance,
                            "rank": i + 1
        
                        })
                        
                print(f"Retrieved {len(retrieved_docs)} documents (after filtering)")
            else:
                print("No documents retrieved from the vector store.")
            
            return retrieved_docs
        
        except Exception as e:
            print(f"Error during retrieval: {e}")
            return []    
                
rag_retriver=RAGRetriver(vectorstore, Embedding_manager)
rag_retriver
test=rag_retriver.retrieve("What is the name of the Hotel in Tokyo")
print(test)

from langchain_groq import ChatGroq
import os
from dotenv import load_dotenv
load_dotenv()

groq_api_key = ("")
llm = ChatGroq(api_key=groq_api_key, model_name = "llama-3.3-70b-versatile", temperature=0.1, max_tokens=1024)

## Simple RAG Function

def rag_simple(query, retriver, llm, top_k=3):
    # Step 1: Retrieve relevant documents
    #retrieved_docs = retriver.retrieve(query, top_k=top_k):
    results = retriver.retrieve(query, top_k=top_k)
    context = "\n\n".join([doc['content'] for doc in results]) if results else ""
    
    if not context:
        return "No relevant documents found."
    
    ## Step 2: Generate response using LLM
    prompt = """Use the following context to answer the question concisely.
        Context:
        {context}
    
        Question: {query}
        Answer:"""
        
    response = llm.invoke([prompt.format(context=context, query=query)])
    return response.content

answer=rag_simple("What is the name of the Hotel in Tokyo?", rag_retriver, llm)
print(answer)

def rag_advanced(query, retriver, llm, top_k=5, min_score=0.2, return_context=False):
    
    # Step 1: Retrieve relevant documents
    results = retriver.retrieve(query, top_k=top_k, score_threshold=min_score)
    if not results:
        return {'answer': 'No relevant context found.', 'source': [], 'confidence': 0.0, 'context': ''}

    context = "\n\n".join([doc['content'] for doc in results])
    sources = [{
        'source': doc['metadata'].get('source_file', doc['metadata'].get('source', 'unknown')),
        'page': doc['metadata'].get('page', 'unknown'),
        'score': doc['similarity_score'],
        'preview': doc['content'][:300] + '...'
    } for doc in results]

    confidence = max([doc['similarity_score'] for doc in results])
            
    ## Step 2: Generate response using LLM
    prompt = f"""Use the following context to answer the question concisely.
Context:
{context}

Question: {query}
Answer:"""
        
    response = llm.invoke([prompt])
    
    output = {
        'answer': response.content,
        'source': sources,
        'confidence': confidence
    }
    if return_context:
        output['context'] = context
    return output
result = rag_advanced("What is the name of the Hotel in Osaka?", rag_retriver, llm, top_k=5, min_score=0.2, return_context=True)
print("Answer:", result['answer'])
print("Sources:", result['source'])
print("Confidence:", result['confidence'])
print("Context:", result['context'][:300])

# class AdvancedRAGPipeline:
#     def __init__(self, retriver, llm):
#         self.retriver = retriver
#         self.llm = llm
#         self.history = []
#     def query(self, question: str, top_k=5, min_score=0.2, stream: bool = False, summarize: bool = False):
#         pass

print("OK")