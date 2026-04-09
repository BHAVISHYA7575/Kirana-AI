#IMPORTS
from langgraph.graph import StateGraph, END
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
import os
from dotenv import load_dotenv
load_dotenv()
import faiss
import pickle
from sentence_transformers import SentenceTransformer
from typing import TypedDict, List 

#MODEL
model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')


#DATA
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

index = faiss.read_index(os.path.join(BASE_DIR, "embeddings", "store.index"))
with open(os.path.join(BASE_DIR, "embeddings", "index.pkl"), 'rb') as f:
    all_texts = pickle.load(f)

#INITIALIZE LANGUAGE MODEL
llm = ChatGroq(api_key=os.getenv("GROQ_API_KEY"), model_name="llama-3.3-70b-versatile")


#DEFINE STATE: INFO. SHARED BETWEEN NODES
class AgentState(TypedDict):
    query: str
    retrieved_docs: List[str]
    response: str

#RETRIEVE FUNCTION
def retrieve(query):
    query_vector = model.encode(query)
    query_vector = query_vector.reshape(1, -1)
    #FIND THE MOST SIMILAR TEXTS FROM THE EMBEDDING STORE 
    distances,indices = index.search(query_vector,3)
    results = [all_texts[i] for i in indices[0]]
    return results 

#NODES OF THE LANGRAPH STARTS 

#RETRIEVE NODE:TAKES QUERY AND RETURNS RELEVANT INFORMATION 
def retrieve_node(state: AgentState):
    query = state["query"]
    docs = retrieve(query)
    return {"retrieved_docs": docs}

#RESPONSE NODE: TAKES QUERY AND RETRIEVED DOCS TO GENERATE RESPONSE WITH THE HELP OF LANGUAGE MODEL
def generate_response_node(state: AgentState):
    query = state["query"]
    docs = state["retrieved_docs"]
    context = "\n".join(docs)
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant for a kirana store owner in India. Answer based only on the provided context. Reply in simple Hindi or English based on the question."),
        ("human", f"Context:\n{context}\n\nQuestion: {query}")
    ])
    
    chain = prompt | llm
    response = chain.invoke({})
    
    return {"response": response.content}


#BUILD THE AGENT GRAPH 
workflow = StateGraph(AgentState)

workflow.add_node("retrieve", retrieve_node)
workflow.add_node("generate", generate_response_node)

workflow.set_entry_point("retrieve")
workflow.add_edge("retrieve", "generate")
workflow.add_edge("generate", END)

app = workflow.compile()

# TEST THE AGENT
result = app.invoke({"query": "sarso ka tel kitna bacha hai", "retrieved_docs": [], "response": ""})
print(result["response"])