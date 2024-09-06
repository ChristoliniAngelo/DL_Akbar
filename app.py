import cv2
import numpy as np
import tensorflow as tf
import time
import threading
import sys
from tkinter import Tk, Label, Button, StringVar, OptionMenu

# Cek versi library
print(f"Versi OpenCV: {cv2.__version__}")
print(f"Versi NumPy: {np.__version__}")
print(f"Versi TensorFlow: {tf.__version__}")
print(f"Versi Python: {sys.version}")

# Fungsi untuk memproses frame dari webcam
def preprocess_image(frame, input_size=(224, 224)):
    img_resized = cv2.resize(frame, input_size)  # Ubah ukuran sesuai input model
    img_array = np.array(img_resized)
    img_array = np.expand_dims(img_array, axis=0)  # Tambahkan batch dimension
    img_array = img_array / 255.0  # Normalisasi
    return img_array

# Fungsi untuk memetakan prediksi ke label dengan nilai akurasi
def classify_tomato(prediction, threshold=0.6):
    labels = ['belum matang', 'matang']  # Label yang digunakan
    predicted_index = np.argmax(prediction)
    predicted_label = labels[predicted_index]
    confidence = prediction[0][predicted_index]  # Ambil nilai probabilitas untuk label yang diprediksi
    
    # Jika confidence di bawah threshold, hasil dianggap tidak pasti
    if confidence < threshold:
        predicted_label = 'Tidak Yakin'
    
    return predicted_label, confidence

# Fungsi untuk menangani prediksi secara real-time
def predict_frame(model, frame, prediction_queue, history, stable_count=5):
    preprocessed_img = preprocess_image(frame)
    prediction = model.predict(preprocessed_img)
    predicted_label, confidence = classify_tomato(prediction)

    # Tambahkan ke history untuk stabilisasi hasil
    history.append(predicted_label)
    
    # Hanya hasilkan prediksi stabil setelah beberapa prediksi berturut-turut
    if len(history) > stable_count:
        history.pop(0)  # Buang prediksi terlama
        if all(h == predicted_label for h in history):
            prediction_queue.append((predicted_label, confidence))
        else:
            prediction_queue.append(('Tidak Stabil', 0.0))

# Fungsi untuk menampilkan FPS
def display_fps(start_time):
    end_time = time.time()
    fps = 1 / (end_time - start_time)
    return fps

# Fungsi utama yang memulai kamera dan klasifikasi
def start_classification(webcam_index):
    # Muat model
    model = tf.keras.models.load_model('model.h5')

    # Inisialisasi kamera
    cap = cv2.VideoCapture(int(webcam_index))
    if not cap.isOpened():
        print("Kamera tidak dapat diakses.")
        return

    prediction_queue = []
    history = []  # Untuk menyimpan prediksi sebelumnya
    thread = None

    while True:
        start_time = time.time()
        ret, frame = cap.read()
        if not ret:
            print("Gagal menangkap gambar.")
            break

        # Mulai thread prediksi
        if thread is None or not thread.is_alive():
            thread = threading.Thread(target=predict_frame, args=(model, frame, prediction_queue, history))
            thread.start()

        # Tampilkan prediksi jika ada
        if prediction_queue:
            label, confidence = prediction_queue.pop(0)
        else:
            label = "Tidak Ada Tomat"
            confidence = 0.0

        # Gambar kotak di sekitar area tomat
        height, width, _ = frame.shape
        box_color = (0, 255, 0) if label == "matang" else (0, 0, 255)
        cv2.rectangle(frame, (50, 50), (width-50, height-50), box_color, 2)

        # Tampilkan prediksi dan akurasi di frame
        cv2.putText(frame, f'Prediksi: {label}', (60, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, box_color, 2)
        cv2.putText(frame, f'Akurasi: {confidence:.2f}', (60, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, box_color, 2)

        # Tampilkan FPS
        fps = display_fps(start_time)
        cv2.putText(frame, f'FPS: {fps:.2f}', (60, 140), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        # Tampilkan hasil
        cv2.imshow('Klasifikasi Tomat Real-Time', frame)

        # Keluar jika menekan 'q' atau jika jendela ditutup
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or cv2.getWindowProperty('Klasifikasi Tomat Real-Time', cv2.WND_PROP_VISIBLE) < 1:
            print("Menutup aplikasi...")
            break

    # Tutup kamera dan jendela
    cap.release()
    cv2.destroyAllWindows()
    sys.exit(0)  # Akhiri proses

# Fungsi untuk memulai klasifikasi setelah memilih webcam dari UI
def start_ui():
    root = Tk()
    root.title("Pilih Webcam")

    # Atur ukuran window dan posisinya di tengah layar
    window_width = 300
    window_height = 150

    # Dapatkan ukuran layar
    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()

    # Hitung posisi x dan y untuk menempatkan window di tengah
    x_cordinate = int((screen_width/2) - (window_width/2))
    y_cordinate = int((screen_height/2) - (window_height/2))

    # Atur ukuran dan posisi window
    root.geometry(f"{window_width}x{window_height}+{x_cordinate}+{y_cordinate}")
    root.resizable(False, False)  # Nonaktifkan resize window

    # Pilihan webcam (0 untuk internal, 1 untuk eksternal)
    webcam_var = StringVar(root)
    webcam_var.set("0")  

    # Label dan dropdown untuk memilih webcam
    label = Label(root, text="Pilih Webcam", font=("Arial", 12))
    label.pack(pady=10)

    option_menu = OptionMenu(root, webcam_var, "0", "1")
    option_menu.config(width=15, font=("Arial", 10))
    option_menu.pack(pady=5)

    # Tombol untuk memulai klasifikasi
    start_button = Button(root, text="Mulai Klasifikasi", command=lambda: [root.destroy(), start_classification(webcam_var.get())], width=20)
    start_button.pack(pady=10)

    root.mainloop()

# Menjalankan UI jika program dijalankan langsung
if __name__ == "__main__":
    start_ui()
