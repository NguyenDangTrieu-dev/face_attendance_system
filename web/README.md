📅 ỨNG DỤNG ĐIỂM DANH KHUÔN MẶT REAL-TIME QUA MẠNG LAN

🚀 Giới Thiệu

Ứng dụng điểm danh khuôn mặt real-time hoạt động trên nền tảng Web và Mobile, cho phép kết nối các thiết bị cùng mạng LAN để nhận diện và điểm danh nhanh chóng, chính xác.

🧠 Yêu Cầu Hệ Thống

-**Python 3.10+**

-**Docker Desktop**-

**Thiết bị di động Android (app đã đóng gói sẵn file APK)**

**Tất cả thiết bị cần kết nối chung mạng LAN**

# 🛠️ Cài Đặt Backend

Bước 1: Tải và Cài Docker

Truy cập: https://www.docker.com/products/docker-desktop

Kiểm tra sau khi cài đặt:

docker --version

Bước 2: Tải Mã Nguồn và giải nén



Bước 3: Khởi Động Database và Backend Service

docker compose up -d

Lệnh này sẽ khởi chạy database và các dịch vụ nền tảng cần thiết.

Bước 4: Cài Thư Viện Python
    -**Mở Terminal nhập:

        pip install -r requirements.txt

Bước 5: Khởi Chạy Ứng Dụng Backend

python src/api/main.py

Bước 6: Truy Cập Giao Diện Web

**Thiết bị khác trong cùng mạng LAN:**



**📱 Cài Đặt App Mobile (File APK)**

Bước 1: Tải File APK

Chép file .apk sang điện thoại Android

Bước 2: Cài App

Cho phép cài đặt từ nguồn không xác định

Mở file .apk để cài đặt

Bước 3: mở APP xin QR của server để đăng nhập vào 

Trong app, khi đăng nhập vào sẽ hiển thị thông tin khóa học.

✅ Hoàn Tất

Hệ thống điểm danh khuôn mặt đã sẵn sàng sử dụng:

✅ Backend & Database

✅ Web App

✅ App Mobile (APK)

*********Tất cả thiết bị cần cùng kết nối trong một mạng LAN để hoạt động chính xác.**************

📩 Liên Hệ / Hỗ Trợ

Tác giả: [Nguyễn Đăng Triều]

Email: [trieutech11@gmail.com]


