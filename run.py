#!/usr/bin/env python3
"""
Script để chạy ứng dụng Face Recognition
"""

import os
import sys

def main():
    # Kiểm tra Python version
    if sys.version_info < (3, 8):
        print("Yêu cầu Python 3.8 trở lên")
        sys.exit(1)
    
    # Kiểm tra các thư viện cần thiết
    try:
        import flask
        import cv2
        import numpy
        import insightface
    except ImportError as e:
        print(f"Thiếu thư viện: {e}")
        print("Vui lòng chạy: pip install -r requirements.txt")
        sys.exit(1)
    
    # Chạy ứng dụng
    from app import app
    print("Đang khởi động ứng dụng Face Recognition...")
    print("Truy cập: http://localhost:5000")
    app.run(debug=True, host='0.0.0.0', port=5000)

if __name__ == '__main__':
    main()
