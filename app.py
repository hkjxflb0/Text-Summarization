from fastapi import FastAPI,Form,Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from heapq import nlargest
import re

app = FastAPI()

templates = Jinja2Templates(directory='templates')

def summarize(text,num_sentence=3)->str:
    if not text.strip():
        return "Input is empty"
    if len(text.split()) < 20:
        return ("So your text is already short no need"
                " to summarize it anymore")
    nlp = spacy.load("en_core_web_sm")
    doc = nlp(text)
    sentences = [sent.text.strip() for sent in doc.sents]

    if len(sentences) == 0:
        return ("No valid sentences found? "
                "do you know english "
                "or your keyboard is not working fine?")
    def clean_sentence(sentence):
        return re.sub(r'\s+', ' ', re.sub(r'[^\w\s]', '', sentence)).lower()

    cleaned_sentences = [clean_sentence(sent) for sent in sentences]

    tfidf_vector = TfidfVectorizer(stop_words='english')
    tfidf_mat = tfidf_vector.fit_transform(cleaned_sentences)
    sentence_score = tfidf_mat.sum(axis=1).flatten().tolist()[0]

    sentence_score_map = {sentences[i]: sentence_score[i] for i in range(len(sentences))}
    top_sentences = nlargest(num_sentence, sentence_score_map, key=sentence_score_map.get)
    summary = " ".join(top_sentences)

    return summary

#Fastapi section
@app.get('/', response_class=HTMLResponse)
def read_root(request: Request):
    return templates.TemplateResponse('index.html',{"request":request})

@app.post("/summarize", response_class = HTMLResponse)
def summaring(request:Request,text:str = Form(...)):
    summary =  summarize(text)
    return templates.TemplateResponse("index.html",{"request":request,"summary":summary})
