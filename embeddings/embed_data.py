#IMPORTS
import json 
import os 
from sentence_transformers import SentenceTransformer
import faiss
import pickle 

#DATA
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")

#MODEL
model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

#LOADING DATA 
def load_json(filename):
    filepath = os.path.join(DATA_DIR, filename)
    with open(filepath,'r', encoding='utf-8') as f:
        return json.load(f)
    
stock_data = load_json('stock.json')
suppliers_data = load_json ('suppliers.json')
bills_data = load_json('bills.json')


#PREPAIRING TEXT
def prepare_stock_text(item):
    return f"Item: {item['item_name']}. Quantity: {item['quantity_kg']} kg. Purchase price: {item['purchase_price']} rupees. Selling price: {item['selling_price']} rupees. Minimum threshold: {item['min_threshold_kg']} kg. Average daily sales: {item['avg_daily_sales_kg']} kg per day."

def prepare_supplier_text(supplier):
    items = ', '.join(supplier['items_supplied'])
    return f"Supplier: {supplier['supplier_name']} supplies {items}. Contact: {supplier['contact_number']}. Lead time: {supplier['lead_time_days']} days. Minimum order: {supplier['min_order_quantgit ity']} kg"

def prepare_bill_text(bill):
    items = ', '.join([f"{i['quantity']} kg {i['name']} at {i['price']} rupees" for i in bill['items']])
    return f"Bill {bill['bill_number']} on {bill['date']}: {items}. Total: {bill['total_amount']} rupees."

#LOOPING THROUGH ITEM TO PREPARE TEXT
stock_texts = [prepare_stock_text(item) for item in stock_data]
supplier_texts = [prepare_supplier_text(supplier) for supplier in suppliers_data]
bill_texts = [prepare_bill_text(bill) for bill in bills_data]

all_texts = stock_texts + supplier_texts + bill_texts

print(f"Total texts to embed: {len(all_texts)}")
print("Sample text:", all_texts[0])

#APPLYING EMBEDDING MODEL
embeddings = model.encode(all_texts, show_progress_bar=True)
print(f"Embeddings created: {embeddings.shape}")

#STORING VECTORS AND ORIGINAL SENTENCES TO EMBEDDINGS
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(embeddings)

faiss.write_index(index, os.path.join(BASE_DIR, "embeddings", "store.index"))

with open(os.path.join(BASE_DIR, "embeddings", "index.pkl"), 'wb') as f :
    pickle.dump(all_texts, f)

print("FAISS index and texts saved successfully.")

