import streamlit as st
import sqlite3
import pandas as pd

from datetime import datetime
from transformers import pipeline
from underthesea import word_tokenize

# --- CẤU HÌNH TRANG ---
st.set_page_config(page_title="Trợ lý phân loại cảm xúc")
st.title("Trợ lý phân loại cảm xúc Tiếng Việt")

# --- CẤU HÌNH TỪ ĐIỂN ---
TEENCODE_DICT = {
    "rat": "rất", "hom": "hôm", "nay": "nay", "dc": "được", 
    "ko": "không", "dở": "tệ", "ok": "tốt", "bt": "bình thường"
}

# --- DATABASE (SQLITE) ---
# Hàm tạo kết nối và bảng nếu chưa có
def init_db():
    conn = sqlite3.connect('history.db')
    c = conn.cursor()
    # Tạo bảng sentiments với 4 cột 
    c.execute('''
        CREATE TABLE IF NOT EXISTS sentiments (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            text TEXT,
            sentiment TEXT,
            timestamp TEXT
        )
    ''')
    conn.commit()
    conn.close()

# Hàm lưu kết quả (Dùng Parameterized Query chống SQL Injection)
def save_to_db(text, sentiment):
    conn = sqlite3.connect('history.db')
    c = conn.cursor()
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S") # Định dạng ISO
    c.execute('INSERT INTO sentiments (text, sentiment, timestamp) VALUES (?, ?, ?)', (text, sentiment, timestamp))
    conn.commit()
    conn.close()

# Hàm lấy lịch sử (Lấy 50 dòng mới nhất)
def load_history():
    conn = sqlite3.connect('history.db')
    c = conn.cursor()
    c.execute('SELECT timestamp, text, sentiment FROM sentiments ORDER BY id DESC LIMIT 50')
    data = c.fetchall()
    conn.close()
    return data

# Hàm lấy lịch sử (Lấy toàn bộ lịch sử)
def load_history_all():
    conn = sqlite3.connect('history.db')
    c = conn.cursor()
    c.execute('SELECT timestamp, text, sentiment FROM sentiments ORDER BY id DESC')
    data = c.fetchall()
    conn.close()
    return data

# Gọi hàm khởi tạo DB ngay khi chạy app
init_db()

# --- CẤU HÌNH AI ---
@st.cache_resource
def load_sentiment_model():
    model_name = "wonrax/phobert-base-vietnamese-sentiment"
    classifier = pipeline("sentiment-analysis", model=model_name)
    return classifier

with st.spinner('Đang khởi động...'):
    classifier = load_sentiment_model()

# --- XỬ LÝ LOGIC ---
def preprocess_text(text):
    if len(text) > 50: return None, "Lỗi: Câu quá dài (> 50 ký tự)!"
    if len(text) < 5: return None, "Lỗi: Câu quá ngắn (< 5 ký tự)!"

    text = text.lower()
    words = text.split()
    corrected_words = [TEENCODE_DICT.get(word, word) for word in words]
    text = " ".join(corrected_words)

    try:
        text_processed = word_tokenize(text, format="text")
    except:
        text_processed = text 
        
    return text_processed, None

def analyze_sentiment(text_processed):
    result = classifier(text_processed)
    raw_label = result[0]['label']
    score = result[0]['score']
    # Nếu điểm thấp, coi như trung lập
    if score < 0.5:
        raw_label = "NEUTRAL"

    # Map các nhãn model viết tắt sang từ rõ nghĩa
    label_map = {
        'POS': 'POSITIVE',
        'NEG': 'NEGATIVE',
        'NEU': 'NEUTRAL',
    }

    mapped_label = label_map.get(str(raw_label).upper(), str(raw_label))
    return mapped_label, score

# --- GIAO DIỆN NGƯỜI DÙNG ---
user_input = st.text_input("Nhập câu tiếng Việt:", placeholder="Ví dụ: Hôm nay tôi rất vui")

if st.button("Phân loại cảm xúc"):
    if user_input:
        text_clean, error = preprocess_text(user_input)
        
        if error:
            st.error(error)
        else:
            # Phân loại
            label, score = analyze_sentiment(text_clean)
            
            # Lưu vào Database
            save_to_db(user_input, label)
            
            # Hiển thị JSON 
            st.json({"text": user_input, "sentiment": label})
            st.toast("Đã lưu vào lịch sử!")

            # Hiển thị kết quả
            st.info(f"Độ tin cậy: {round(score*100, 2)}%")
            
# --- HIỂN THỊ LỊCH SỬ ---
st.divider()
st.subheader("Lịch sử phân loại (50 tin gần nhất)")

# Khởi tạo dataframe vào session state (giữ 1 bảng ổn định trên giao diện)
if 'history_table' not in st.session_state:
    history_data = load_history()
    if history_data:
        st.session_state['history_table'] = pd.DataFrame(history_data, columns=["Thời gian", "Nội dung", "Cảm xúc"])
    else:
        st.session_state['history_table'] = pd.DataFrame(columns=["Thời gian", "Nội dung", "Cảm xúc"])

# Placeholder cho bảng - tạo 1 phần tử trong layout
table_placeholder = st.empty()

if not st.session_state['history_table'].empty:
    table_placeholder.dataframe(st.session_state['history_table'], use_container_width=True)
else:
    st.info("Chưa có dữ liệu lịch sử.")

# Hai nút - cập nhật (tải lại 50 dòng gần nhất) và hiển thị toàn bộ
col1, col2 = st.columns([1, 1])
with col1:
    if st.button("Cập nhật danh sách"):
        # Tải lại 50 dòng gần nhất và thay thế session dataframe
        history_data = load_history()
        if history_data:
            st.session_state['history_table'] = pd.DataFrame(history_data, columns=["Thời gian", "Nội dung", "Cảm xúc"])
        else:
            st.session_state['history_table'] = pd.DataFrame(columns=["Thời gian", "Nội dung", "Cảm xúc"])
        table_placeholder.dataframe(st.session_state['history_table'], use_container_width=True)

with col2:
    if st.button("Hiển thị toàn bộ lịch sử"):
        # Lấy toàn bộ lịch sử
        all_data = load_history_all()
        if all_data:
            df_all = pd.DataFrame(all_data, columns=["Thời gian", "Nội dung", "Cảm xúc"])
            # Append các dòng mới không có trong dataframe hiện tại
            if st.session_state['history_table'].empty:
                st.session_state['history_table'] = df_all
            else:
                combined = pd.concat([st.session_state['history_table'], df_all], ignore_index=True)
                # Loại bỏ bản ghi trùng lặp (giữ bản ghi đã có trước đó)
                combined = combined.drop_duplicates(keep='first').reset_index(drop=True)
                st.session_state['history_table'] = combined
            table_placeholder.dataframe(st.session_state['history_table'], use_container_width=True)
        else:
            st.info("Chưa có dữ liệu lịch sử.")

