---
title: Kirana AI
emoji: 🛒
colorFrom: yellow
colorTo: green
sdk: streamlit
sdk_version: 1.28.1
app_file: app.py
pinned: false
---

Problem : A Kirana store in india struggles with the memory and paper by the time they notice a problem, the loss has already happened they need technology advancement in thier business they need a smart AI system.A smart AI system can reduce their losses and maximize their profits and even leads to busines growth and a room for new opportunities. 

tech stack 
1. model : SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
2. Language: Python 
3. Large Language Model : llama-3.3-70b-versatile
4. Vector Database : FAISS
5. Agent Framework : LangGraph
6. User interface : streamlit

Done
phase 1 : Collecting the data and storing it in json files
phase 2 : Converting raw data into vectors for higher retrieval
phase 3 : converting query into vector then retrieving the data and using llm to create a adequate response.
phase 4 : Build a User interface for the Shop owner for using Agentic AI system 

upcoming 
phase 5 : deployment