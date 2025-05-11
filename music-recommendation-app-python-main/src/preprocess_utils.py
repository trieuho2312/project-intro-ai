import re

def simple_clean_text(text):
    """
    Làm sạch văn bản đơn giản:
    - bỏ ký tự không phải chữ
    - tách từ
    - bỏ từ quá ngắn
    """
    text = str(text).lower()
    text = re.sub(r"[^a-z\\s]", "", text)
    words = text.split()
    words = [word for word in words if len(word) > 2]
    return " ".join(words)
