#IMPORTS
import streamlit as st 
import sys 
import os 
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from agent.Kirana_agent import app


#SET PAGE CONFIG
st.set_page_config(page_title="Kirana AI - Your Smart Store Assistant", page_icon="🛒")
st.title("Kirana AI 🛒")
st.subheader("Apni dukan ka smart assistant")
query = st.text_input("Apna sawaal likhein...")
submit = st.button("Poochho")


#INVOKE AGENT WITH SHOPKEEPER QUERY AND DISPLAY RESPONSE 
if submit and query:
    with st.spinner("Soch raha hoon..."):
     result = app.invoke({"query": query, "retrieved_docs": [], "response": ""})
     st.write(result["response"])

