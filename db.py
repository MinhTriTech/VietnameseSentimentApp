import sqlite3
from datetime import datetime

DB_FILE = "history.db"

# DATABASE (SQLITE)
# Hàm tạo kết nối và bảng nếu chưa có
def init_db():
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    # Tạo bảng sentiments với 4 cột 
    c.execute("""
        CREATE TABLE IF NOT EXISTS sentiments (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            text TEXT,
            sentiment TEXT,
            timestamp TEXT
        )
    """)
    conn.commit()
    conn.close()

# Hàm lưu kết quả (Dùng Parameterized Query chống SQL Injection)
def save_to_db(text, sentiment):
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    c.execute("INSERT INTO sentiments (text, sentiment, timestamp) VALUES (?, ?, ?)", (text, sentiment, timestamp))
    conn.commit()
    conn.close()

# Hàm lấy lịch sử (Lấy 50 dòng mới nhất)
def load_history(limit=50):
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute(f"SELECT timestamp, text, sentiment FROM sentiments ORDER BY id DESC LIMIT {limit}")
    data = c.fetchall()
    conn.close()
    return data

# Hàm lấy lịch sử (Lấy toàn bộ lịch sử)
def load_history_all():
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("SELECT timestamp, text, sentiment FROM sentiments ORDER BY id DESC")
    data = c.fetchall()
    conn.close()
    return data
