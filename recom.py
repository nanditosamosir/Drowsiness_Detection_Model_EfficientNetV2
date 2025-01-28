import tensorflow as tf

# Muat model Keras asli
model = tf.keras.models.load_model('Model/efficientnetv2_drowsiness_model_dual_attention.keras')

# Konversi ke format TFLite tanpa optimasi
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = []  # Nonaktifkan semua optimasi
tflite_model = converter.convert()

# Simpan model yang telah dikonversi
with open('efficientnetv2_drowsiness_model_dual_attention_no_optim.tflite', 'wb') as f:
    f.write(tflite_model)
