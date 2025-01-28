import cv2
import tensorflow as tf
from tensorflow.keras.applications.efficientnet_v2 import preprocess_input
import numpy as np

# Muat model yang sudah dilatih
model = tf.keras.models.load_model('Model/efficientnetv2_drowsiness_model_attention.h5')

# Fungsi untuk memproses frame dan membuat prediksi
def predict_frame(frame):
    # Ubah ukuran frame agar sesuai dengan input model
    img = cv2.resize(frame, (260, 260))
    img_array = np.expand_dims(img, axis=0)  # Tambahkan dimensi batch
    img_array = preprocess_input(img_array)  # Praproses gambar

    # Prediksi frame
    prediction = model.predict(img_array)
    return "open" if prediction > 0.5 else "closed"

# Inisialisasi kamera
cap = cv2.VideoCapture(0)  # 0 untuk kamera bawaan

# Loop untuk real-time prediksi
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Buat prediksi pada frame
    predicted_class = predict_frame(frame)

    # Tampilkan hasil prediksi pada frame
    label = f"Status: {predicted_class}"
    cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Tampilkan frame
    cv2.imshow('Real-time Drowsiness Detection', frame)

    # Tekan 'q' untuk keluar
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Tutup kamera dan jendela
cap.release()
cv2.destroyAllWindows()
