import tkinter as tk
from tkinter import Label, Listbox, Button, MULTIPLE
import cv2
from PIL import Image, ImageTk
from ultralytics import YOLO
import pygame
import threading
import time

# =========================
# تهيئة pygame للصوت
# =========================
pygame.mixer.init()
pygame.mixer.music.load(r".\s.mp3")

# =========================
# تحميل الموديل
# =========================
model = YOLO(r".\fire.pt")
class_names = model.names

# =========================
# متغيرات تحكم
# =========================
alarm_playing = False
last_alarm_time = 0
ALARM_COOLDOWN = 5  # ثواني
selected_classes = []

# =========================
# تشغيل الإنذار
# =========================
def play_alarm():
    global alarm_playing, last_alarm_time
    current_time = time.time()

    if not alarm_playing and (current_time - last_alarm_time) > ALARM_COOLDOWN:
        alarm_playing = True
        last_alarm_time = current_time
        pygame.mixer.music.play()
        while pygame.mixer.music.get_busy():
            time.sleep(0.1)
        alarm_playing = False

# =========================
# واجهة المستخدم
# =========================
root = tk.Tk()
root.title("Fire Detection System")
root.geometry("1150x700")
root.configure(bg="#1e1e1e")

Label(
    root,
    text="🔥 Fire Detection System",
    font=("Arial", 22, "bold"),
    fg="orange",
    bg="#1e1e1e"
).pack(pady=10)

Label(
    root,
    text="Select classes to trigger alarm:",
    font=("Arial", 14),
    fg="white",
    bg="#1e1e1e"
).pack(pady=5)

class_listbox = Listbox(
    root,
    selectmode=MULTIPLE,
    font=("Arial", 13),
    width=30,
    height=6
)
class_listbox.pack(pady=5)

for idx, name in class_names.items():
    class_listbox.insert(tk.END, f"{idx} - {name}")

# =========================
# إطار الفيديو
# =========================
video_frame = tk.Frame(root, bg="#2b2b2b", bd=3)
video_frame.pack(pady=10)

video_label = Label(video_frame)
video_label.pack()

status_label = Label(
    root,
    text="Status: Waiting...",
    font=("Arial", 18, "bold"),
    fg="white",
    bg="#1e1e1e"
)
status_label.pack(pady=10)

cap = cv2.VideoCapture(0)

# =========================
# تحديث الإطارات
# =========================
def update_frame():
    ret, frame = cap.read()
    if not ret:
        return

    results = model(frame, verbose=False)
    detected_alarm = False

    for r in results:
        for box in r.boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])

            if cls_id in selected_classes and conf > 0.5:
                detected_alarm = True

    annotated = results[0].plot()
    annotated = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)

    img = Image.fromarray(annotated).resize((900, 450))
    imgtk = ImageTk.PhotoImage(image=img)

    video_label.imgtk = imgtk
    video_label.configure(image=imgtk)

    if detected_alarm:
        status_label.config(text="🚨 DANGER DETECTED!", fg="red")
        threading.Thread(target=play_alarm, daemon=True).start()
    else:
        status_label.config(text="✅ SAFE", fg="green")

    root.after(15, update_frame)

# =========================
# زر البدء
# =========================
def start_detection():
    global selected_classes
    selected_classes = [
        int(class_listbox.get(i).split(" - ")[0])
        for i in class_listbox.curselection()
    ]
    update_frame()

Button(
    root,
    text="▶ Start Detection",
    font=("Arial", 14, "bold"),
    bg="orange",
    fg="black",
    width=20,
    command=start_detection
).pack(pady=10)

root.mainloop()

cap.release()
cv2.destroyAllWindows()
pygame.mixer.quit()
