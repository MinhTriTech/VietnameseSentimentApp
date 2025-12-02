import streamlit as st
import pandas as pd
from datetime import datetime

from db import init_db, save_to_db, load_history, load_history_all
from model import get_classifier, preprocess_text, analyze_sentiment

# Cấu hình trang 
st.set_page_config(page_title="Trợ lý phân loại cảm xúc")
st.title("Trợ lý phân loại cảm xúc Tiếng Việt")

# Constants
HISTORY_COLUMNS = ["Thời gian", "Nội dung", "Cảm xúc"]
HISTORY_KEY = "history_table"

# Khởi tạo database 
init_db()

# Load model 
classifier = get_classifier()  # trong model.py đã dùng cache + spinner

# Tạo dataFrame từ list data hoặc trả về empty nếu data None
def create_history_df(data):
    return pd.DataFrame(data, columns=HISTORY_COLUMNS) if data else pd.DataFrame(columns=HISTORY_COLUMNS)

# Cập nhật session_state history_table
def update_session_history(data, append=False):
    df_new = create_history_df(data)
    if append and not st.session_state[HISTORY_KEY].empty:
        combined = pd.concat([st.session_state[HISTORY_KEY], df_new], ignore_index=True)
        combined = combined.drop_duplicates(keep="first").reset_index(drop=True)
        st.session_state[HISTORY_KEY] = combined
    else:
        st.session_state[HISTORY_KEY] = df_new

# Xử lý input từ người dùng
user_input = st.text_input("Nhập câu tiếng Việt:", placeholder="Ví dụ: Hôm nay tôi rất vui")

# Xử lý input, phân loại sentiment, lưu DB
def handle_user_input(text):
    if not text:
        return {"error": "Vui lòng nhập câu cần phân loại."}

    text_clean, error = preprocess_text(text)
    if error:
        return {"error": error}

    label, score = analyze_sentiment(text_clean)
    save_to_db(text, label)
    return {"text": text, "label": label, "score": score}

if st.button("Phân loại cảm xúc"):
    result = handle_user_input(user_input)
    if "error" in result:
        st.error(result["error"])
    else:
        st.json({"text": result["text"], "sentiment": result["label"]})
        st.info(f"Độ tin cậy: {round(result['score']*100,2)}%")
        st.toast("Đã lưu vào lịch sử!")
        update_session_history(load_history())

# Hiển thị lịch sử
st.divider()
st.subheader("Lịch sử phân loại (50 tin gần nhất)")

# Khởi tạo session_state nếu chưa có
if HISTORY_KEY not in st.session_state:
    update_session_history(load_history())

# Placeholder cho bảng
table_placeholder = st.empty()
if not st.session_state[HISTORY_KEY].empty:
    df_show = st.session_state[HISTORY_KEY].copy()
    df_show.index = df_show.index + 1
    df_show.index.name = "STT"
    table_placeholder.dataframe(df_show, use_container_width=True)

# Nút cập nhật và hiển thị toàn bộ
col1, col2 = st.columns(2)

with col1:
        if st.button("Cập nhật danh sách"):
            update_session_history(load_history())
            df_show = st.session_state[HISTORY_KEY].copy()
            df_show.index = df_show.index + 1
            df_show.index.name = "STT"
            table_placeholder.dataframe(df_show, use_container_width=True)

with col2:
        if st.button("Hiển thị toàn bộ lịch sử"):
            update_session_history(load_history_all(), append=True)
            df_show = st.session_state[HISTORY_KEY].copy()
            df_show.index = df_show.index + 1
            df_show.index.name = "STT"
            table_placeholder.dataframe(df_show, use_container_width=True)
