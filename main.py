import asyncio
import os
import random
from typing import List, Set, Dict, Any
import urllib.parse

import aiohttp
from bs4 import BeautifulSoup
from fastapi import FastAPI, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import WebBaseLoader
from langchain.chains import RetrievalQA
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.llms import OpenAI
from langchain.schema import Document

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

CDP_DOCS = {
    "Segment": "https://segment.com/docs/?ref=nav",
    "mParticle": "https://docs.mparticle.com/",
    "Lytics": "https://docs.lytics.com/",
    "Zeotap": "https://docs.zeotap.com/home/en-us/"
}

FAISS_INDEX_PATH = "faiss_index"
MAX_CRAWL_DEPTH = 2  
MAX_URLS_PER_CDP = 100  
CONCURRENT_REQUESTS = 5  

USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.1.1 Safari/605.1.15",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/92.0.4515.107 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/90.0.4430.212 Safari/537.36 Edg/90.0.818.66",
    "Mozilla/5.0 (iPhone; CPU iPhone OS 14_6 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.0 Mobile/15E148 Safari/604.1"
]

async def fetch_url_with_aiohttp(session: aiohttp.ClientSession, url: str) -> Dict[str, Any]:
    """Fetch a URL using aiohttp and return its content with BeautifulSoup parsing."""
    try:
       
        headers = {
            "User-Agent": random.choice(USER_AGENTS),
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.5",
            "Referer": urllib.parse.urljoin(url, "/"),
            "DNT": "1",
            "Connection": "keep-alive",
            "Upgrade-Insecure-Requests": "1"
        }
        
        async with session.get(url, headers=headers, timeout=30) as response:
            if response.status != 200:
                return {"success": False, "url": url, "error": f"Status code: {response.status}"}
            
            html = await response.text()
            soup = BeautifulSoup(html, 'html.parser')
            
            for script in soup(["script", "style", "nav", "footer", "header"]):
                script.decompose()
            
            text = soup.get_text(separator="\n", strip=True)
            
            links = []
            for a_tag in soup.find_all('a', href=True):
                href = a_tag['href']
                if href and not href.startswith('#') and not href.startswith('mailto:'):
                    links.append(href)
            
            return {
                "success": True,
                "url": url,
                "text": text,
                "title": soup.title.string if soup.title else url,
                "links": links
            }
    except Exception as e:
        return {"success": False, "url": url, "error": str(e)}

async def fetch_url_with_webbaseloader(url: str) -> Dict[str, Any]:
    """Fallback method using WebBaseLoader when aiohttp encounters 403 errors."""
    try:
        loader = WebBaseLoader(
            url,
            header_template={"User-Agent": random.choice(USER_AGENTS)}
        )
        
        docs = await loader.aload()
        
        if not docs:
            return {"success": False, "url": url, "error": "No content found"}
            
        links = []
        try:
            html = docs[0].page_content
            soup = BeautifulSoup(html, 'html.parser')
            for a_tag in soup.find_all('a', href=True):
                href = a_tag['href']
                if href and not href.startswith('#') and not href.startswith('mailto:'):
                    links.append(href)
        except:
            pass 
            
        return {
            "success": True,
            "url": url,
            "text": docs[0].page_content,
            "title": url,  
            "links": links
        }
    except Exception as e:
        return {"success": False, "url": url, "error": f"WebBaseLoader error: {str(e)}"}

async def fetch_url(session: aiohttp.ClientSession, url: str) -> Dict[str, Any]:
    """Try to fetch URL with aiohttp first, fall back to WebBaseLoader if 403 error."""
    result = await fetch_url_with_aiohttp(session, url)
    
    if not result["success"] and "403" in str(result.get("error", "")):
        print(f" Got 403 for {url}, trying WebBaseLoader fallback...")
        result = await fetch_url_with_webbaseloader(url)
        
    return result

async def crawl_website(base_url: str, cdp_name: str) -> List[Document]:
    """Crawl a website recursively up to MAX_CRAWL_DEPTH, following links."""
    documents = []
    visited_urls = set()
    urls_to_visit = [(base_url, 0)]  
    
    delay = 1.0 
    
    async with aiohttp.ClientSession() as session:
        while urls_to_visit and len(visited_urls) < MAX_URLS_PER_CDP:
            current_batch = urls_to_visit[:CONCURRENT_REQUESTS]
            urls_to_visit = urls_to_visit[CONCURRENT_REQUESTS:]
            
            if not current_batch:
                break
                
            tasks = []
            for url, depth in current_batch:
                if url in visited_urls:
                    continue
                    
                visited_urls.add(url)
                tasks.append(fetch_url(session, url))
            
            if not tasks:
                continue
                
            results = await asyncio.gather(*tasks)
            
            for result in results:
                if not result["success"]:
                    print(f"⚠️ Error fetching {result['url']}: {result.get('error')}")
                    continue
                
                print(f" Crawled: {result['url']}")
                
                doc = Document(
                    page_content=result["text"],
                    metadata={
                        "source": result["url"],
                        "title": result["title"],
                        "cdp": cdp_name
                    }
                )
                documents.append(doc)
                
                current_depth = next((depth for u, depth in current_batch if u == result["url"]), 0)
                if current_depth < MAX_CRAWL_DEPTH:
                    for link in result["links"]:
                        full_url = urllib.parse.urljoin(result["url"], link)
                        
                        if urllib.parse.urlparse(base_url).netloc == urllib.parse.urlparse(full_url).netloc:
                            if full_url not in visited_urls:
                                urls_to_visit.append((full_url, current_depth + 1))
            
            await asyncio.sleep(delay)
    
    print(f" Crawled {len(documents)} pages from {cdp_name}")
    return documents

async def load_documents() -> List[Document]:
    """Loads documentation pages asynchronously from all CDPs."""
    all_documents = []
    
    for cdp, url in CDP_DOCS.items():
        try:
            print(f" Starting crawl of {cdp} at {url}")
            cdp_docs = await crawl_website(url, cdp)
            all_documents.extend(cdp_docs)
            print(f" Finished crawling {cdp}: {len(cdp_docs)} documents")
        except Exception as e:
            print(f" Error crawling {cdp}: {e}")
            try:
                print(f" Attempting direct load of {cdp} with WebBaseLoader...")
                loader = WebBaseLoader(
                    url,
                    header_template={"User-Agent": random.choice(USER_AGENTS)}
                )
                docs = await loader.aload()
                for doc in docs:
                    doc.metadata["cdp"] = cdp
                all_documents.extend(docs)
                print(f"Loaded {len(docs)} documents from {cdp} using WebBaseLoader")
            except Exception as e2:
                print(f"Failed to load {cdp} with WebBaseLoader: {e2}")
    
    return all_documents

async def preprocess_documents():
    """Loads, splits, and preprocesses documents before embedding."""
    raw_docs = await load_documents()
    print(f"Processing {len(raw_docs)} total documents")
    
    if not raw_docs:
        print("⚠️ No documents were loaded. Vector store will be empty.")
        return []
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=2000, 
        chunk_overlap=100,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    
    split_docs = text_splitter.split_documents(raw_docs)
    print(f"Created {len(split_docs)} chunks for embedding")
    
    return split_docs

async def create_or_load_vectorstore():
    """Creates FAISS index if not found, else loads from disk."""
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    
    if os.path.exists(FAISS_INDEX_PATH) and os.path.isdir(FAISS_INDEX_PATH):
        try:
            print("Loading FAISS index from disk...")
            vectorstore = FAISS.load_local(FAISS_INDEX_PATH, embeddings, allow_dangerous_deserialization=True)
            print(f"Loaded index with {vectorstore.index.ntotal} vectors")
            return vectorstore
        except Exception as e:
            print(f"Error loading FAISS index: {e}")
            print("Building new index instead...")
    
    print("Loading and processing documents...")
    cleaned_docs = await preprocess_documents()
    
    if not cleaned_docs:
        raise RuntimeError("No documents were loaded. Please check your network connection and CDP URLs.")
    
    print(f"Creating FAISS index from {len(cleaned_docs)} documents...")
    vectorstore = FAISS.from_documents(cleaned_docs, embeddings)
    
    os.makedirs(FAISS_INDEX_PATH, exist_ok=True)
    vectorstore.save_local(FAISS_INDEX_PATH)
    print("FAISS index created and saved successfully!")

    return vectorstore

def load_llm():
    """Loads the LLM via LM Studio."""
    return OpenAI(
        base_url="http://localhost:1234/v1",
        api_key="lm-studio",
        max_tokens=4096,
        temperature=0.7,  
    )

@app.on_event("startup")
async def load_vector_store():
    print("🚀 Initializing vector store and retriever...")
    try:
        vectorstore = await create_or_load_vectorstore()
        retriever = vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 10}
        )

        llm = load_llm()
        
        app.state.vectorstore = vectorstore
        app.state.qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            retriever=retriever,
            return_source_documents=True
        )
        print("Chatbot is ready!")
    except Exception as e:
        print(f"Failed to initialize: {e}")

class QuestionRequest(BaseModel):
    question: str

async def get_qa_chain():
    if not hasattr(app.state, "qa_chain"):
        raise HTTPException(status_code=503, detail="QA chain not initialized. Try again later.")
    return app.state.qa_chain

@app.post("/ask")
async def ask_question(req: QuestionRequest, qa_chain: RetrievalQA = Depends(get_qa_chain)):
    try:
        result = qa_chain({"query": req.question + " Provide a detailed and extensive answer."})
        
        answer = result.get("result", "No answer found")
        sources = []
        
        if "source_documents" in result:
            for doc in result["source_documents"]:
                sources.append({
                    "content": doc.page_content[:200] + "...",  
                    "source": doc.metadata.get("source", "Unknown"),
                    "title": doc.metadata.get("title", "Unknown"),
                    "cdp": doc.metadata.get("cdp", "Unknown")
                })
        
        return {
            "answer": answer,
            "sources": sources
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing question: {str(e)}")

@app.get("/health")
async def health_check():
    """Health check endpoint to verify the service is running."""
    is_ready = hasattr(app.state, "qa_chain")
    return {
        "status": "ready" if is_ready else "initializing",
        "message": "QA system is ready to answer questions" if is_ready else "QA system is still initializing"
    }

@app.get("/crawl_status")
async def crawl_status():
    """Get information about the crawled documents."""
    if hasattr(app.state, "vectorstore"):
        return {
            "status": "ready",
            "total_vectors": app.state.vectorstore.index.ntotal,
            "cdps": list(CDP_DOCS.keys())
        }
    return {
        "status": "initializing",
        "message": "Vector store not yet initialized"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)