import cv2
import tensorflow as tf
from tensorflow.keras.applications.efficientnet_v2 import preprocess_input
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Lambda

# Dekorator untuk memastikan fungsi dapat dikenali saat deserialisasi
@tf.keras.utils.register_keras_serializable()
def reduce_mean_layer(input_tensor):
    return tf.reduce_mean(input_tensor, axis=-1, keepdims=True)

@tf.keras.utils.register_keras_serializable()
def reduce_max_layer(input_tensor):
    return tf.reduce_max(input_tensor, axis=-1, keepdims=True)

# Muat model yang sudah dilatih
model = load_model('Model/efficientnetV2_Model_Dual_Attention.h5', custom_objects={
    'reduce_mean_layer': reduce_mean_layer,
    'reduce_max_layer': reduce_max_layer
})

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