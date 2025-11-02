from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import requests
import numpy as np
import tensorflow as tf
import joblib
import os

app = FastAPI()

SIMILARITY_SEARCH_URL = os.getenv("SIMILARITY_SEARCH_URL")
# templates = Jinja2Templates(directory="templates")

# embedding_model = sentenceTransformer("all-MiniLM-L6-v2")

# def generate_id(text):
#     return hashlib.md5(text.encode()).hexdigest()

# Load model and encoders
class DressAPIModel:

    #chroma db setup
#     def init_chroma():
#         client = chromadb.Client()
#         collection_occasion = client.get_or_create_collection(
#             name= 'occasion',
#             metadata = {"hnsw:space" : "cosine"}
#         )
#         collection_country = client.get_or_create_collection(
#             name= 'country',
#             metadata = {"hnsw:space" : "cosine"}
#         )
#         return collection_occasion, collection_country

#     def add_document(collection_occasion, collection_country):
#         doc_occasion = ['casual_outing', 'picnic', 'graduation', 'beach_party', 'wedding', 'formal_dinner', 'business_meeting', 'religious_event', 'job_interview', 'nightclub', 'cultural_festival']
#         doc_country = ['nigeria', 'france', 'uk', 'uae', 'usa', 'brazil', 'japan', 'germany', 'saudi_arabia', 'canada', 'australia', 'india', 'south_africa', 'china', 'mexico']

#         embeddings_occasion = embedding_model.encode(doc_occasion).tolist()
# # 
#         ids_occasion = [generate_id(doc) for doc in doc_occasion]

#         collection_occasion.add(
#             ids = ids_occasion,
#             documents = doc_occasion,
#             embeddings = embeddings_occasion,
#             metadatas = None
#         )
#         embeddings_country = embedding_model.encode(doc_country).tolist()

#         ids_country = [generate_id(doc) for doc in doc_country]

#         collection_country.add(
#             ids = ids_country,
#             documents = doc_country,
#             embeddings = embeddings_country,
#             metadatas = None
#         )
#         return ids_occasion, ids_country

#     def search_similar(collection, query, n_results=1):
#         query_embeddings = embedding_model.encode([query]).tolist()
#         result = collection.query(
#             query_embeddings = query_embeddings,
#             n_results = n_results
#         )
#         return result

    #communicating with similar search microservice
    def get_similar(self, occasion, country):
        url = SIMILARITY_SEARCH_URL
        payload = {
            "occasion": occasion,
            "country": country
        }
        response = requests.post(url, json=payload, timeout=35)
        if response.status_code==200:
            data = response.json()
            return data['occasion_similar']['documents'][0][0], data['country_similar']['documents'][0][0]
        else:
            raise Exception(f"Failed to fetch from similarity service {response.text}")
    

    #Recommender Model
    def __init__(self, model_path="model/"):
        self.model = tf.keras.models.load_model(os.path.join(model_path, "dress_model.keras"))
        self.encoders = {
            name.split("_encoder.pkl")[0]: joblib.load(os.path.join(model_path, name))
            for name in os.listdir(model_path)
            if name.endswith("_encoder.pkl")
        }
        self.vocab_sizes = {k: len(enc.classes_) for k, enc in self.encoders.items()}

    def predict(self, occasion, country, gender, formality=None, context=None):
        try:
            # Inference defaults
            formality = formality or {
                'business_meeting': 'formal', 'picnic': 'casual',
                'wedding': 'formal', 'job_interview': 'formal',
                'beach_party': 'party', 'nightclub': 'party', 'cultural_festival': 'traditional',
                'graduation': 'semi_formal'
            }.get(occasion, 'casual')

            context = context or {
                'saudi_arabia': 'conservative', 'uae': 'conservative',
                'usa': 'moderate', 'uk': 'moderate', 'brazil': 'relaxed'
            }.get(country, 'moderate')

            gender = gender or 'unisex'

            male_dresses = ['blazer_and_jeans', 'formal_suit', 'tuxedo', 'ethnic_kurta_pajama', 'african_dashiki']
            female_dresses = ['summer_dress', 'midi_dress', 'cocktail_dress', 'evening_gown', 'kimono', 'middle_eastern_kaftan', 'saree', 'lehenga', 'abaya']

            #get the most similar value from chroma db if user enters smthg not alreday present
            # occasion = dress_model.get_similar(occasion)
            # country = dress_model.get_similar(country)

            # Encode
            encoded_input = {
                'occasion': np.array([self.encoders['occasion'].transform([occasion])[0]]),
                'country': np.array([self.encoders['country'].transform([country])[0]]),
                'formality': np.array([self.encoders['occasion_formality'].transform([formality])[0]]),
                'context': np.array([self.encoders['cultural_context'].transform([context])[0]]),
                'gender': np.array([self.encoders['gender'].transform([gender])[0]]),
            }

            dress_labels = self.encoders['dress_recommendation'].classes_  # decoded dress names

            probs = self.model.predict(encoded_input)
           
            probs_flat = probs[0]

            for idx, dress in enumerate(dress_labels):
              if(gender == 'male' and dress in female_dresses) or \
                (gender == 'female' and dress in male_dresses):
                  probs_flat[idx] = 0

            top3_idx = np.argsort(probs[0])[-3:][::-1]
            top3_labels = self.encoders['dress_recommendation'].inverse_transform(top3_idx)
            top3_conf = probs[0][top3_idx]

            return {
                "top_recommendation": top3_labels[0],
                "confidence": float(top3_conf[0]),
                "top_3": [(str(label), float(conf)) for label, conf in zip(top3_labels, top3_conf)]
            }

        except Exception as e:
            return {"error": str(e)}
        
# Instantiate model
dress_model = DressAPIModel()

#init chromadb
# collection_occasion, collection_country = dress_model.init_chroma()
# ids_occasion, ids_country = dress_model.add_document(collection_occasion, collection_country)

# API Input model
class PredictRequest(BaseModel):
    occasion: str
    country: str
    formality: str = None
    context: str = None
    gender: str
    

# JSON prediction endpoint
@app.post("/predict")
def predict(request: PredictRequest):
    occasion, country = dress_model.get_similar(request.occasion, request.country)
    print(occasion)
    print(country)
    return dress_model.predict(
        occasion=occasion,
        country=country,
        formality=request.formality,
        context=request.context,
        gender=request.gender,
    )


