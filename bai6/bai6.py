import cv2

# Tải Haar Cascade
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Hàm nhận diện khuôn mặt
def detect_faces(image_path=None):
    if image_path:
        img = cv2.imread('C:/Users/admin/Downloads/XLABai1/anhnguoi.jpg')  # Đọc ảnh từ tệp
    else:
        cap = cv2.VideoCapture(0)  # Sử dụng webcam
        while True:
            ret, img = cap.read()
            if not ret:
                break

            # Phát hiện khuôn mặt
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

            # Đánh dấu khuôn mặt
            for (x, y, w, h) in faces:
                cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)

            # Hiển thị ảnh
            cv2.imshow('Webcam Face Detection', img)

            # Thoát khi nhấn phím 'q'
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()
        return

    # Nếu có đường dẫn ảnh
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    # Đánh dấu khuôn mặt
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)

    # Hiển thị ảnh
    cv2.imshow('Face Detection', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Gọi hàm với đường dẫn ảnh hoặc None để sử dụng webcam
# detect_faces('path_to_your_image.jpg')  # Đường dẫn ảnh
detect_faces()  # Sử dụng webcam