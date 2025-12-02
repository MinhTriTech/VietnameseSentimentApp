import streamlit as st
from transformers import pipeline
from underthesea import word_tokenize

# Cấu hình từ điển 
TEENCODE_DICT = {
    "rat": "rất", "hom": "hôm", "nay": "nay", "dc": "được", 
    "ko": "không", "dở": "tệ", "ok": "tốt", "bt": "bình thường"
}

# Load model 
@st.cache_resource
def load_sentiment_model():
    model_name = "wonrax/phobert-base-vietnamese-sentiment"
    classifier = pipeline("sentiment-analysis", model=model_name)
    return classifier

def get_classifier():
    with st.spinner("Đang khởi động mô hình..."):
        return load_sentiment_model()

# Xử lý văn bản
def preprocess_text(text):
    if len(text) > 50: return None, "Lỗi: Câu quá dài (>50 ký tự)!"
    if len(text) < 5: return None, "Lỗi: Câu quá ngắn (<5 ký tự)!"

    text = text.lower()
    words = text.split()
    corrected_words = [TEENCODE_DICT.get(word, word) for word in words]
    text = " ".join(corrected_words)

    try:
        text_processed = word_tokenize(text, format="text")
    except Exception as e:
        text_processed = text

    return text_processed, None

def analyze_sentiment(text_processed):
    classifier = get_classifier()  
    result = classifier(text_processed)
    raw_label = result[0]['label']
    score = result[0]['score']

    if score < 0.5:
        raw_label = "NEUTRAL"
    # Map các nhãn model viết tắt sang từ rõ nghĩa
    label_map = {'POS':'POSITIVE', 'NEG':'NEGATIVE', 'NEU':'NEUTRAL'}
    mapped_label = label_map.get(str(raw_label).upper(), str(raw_label))
    return mapped_label, score
