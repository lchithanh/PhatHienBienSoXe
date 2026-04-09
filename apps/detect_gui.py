import threading
import tkinter as tk
from tkinter import filedialog, messagebox
from pathlib import Path
import cv2
from PIL import Image, ImageTk
import sys

sys.path.append(str(Path(__file__).resolve().parent.parent))
from scr.detection.no_plate_engine import NoPlateEngine
from scr.logging.violation_logger import ViolationLogger
import config


class NoPlateGUI:
    def __init__(self, root):
        self.root = root
        self.root.title('NoPlate Detection - Webcam + Image')

        self.engine = NoPlateEngine(enable_ocr=config.ENABLE_OCR)
        self.logger = ViolationLogger()

        self.webcam_running = False
        self.cap = None
        self.current_frame = None

        self.create_widgets()

    def create_widgets(self):
        # left side (webcam)
        left_frame = tk.Frame(self.root)
        left_frame.grid(row=0, column=0, padx=10, pady=10)

        tk.Label(left_frame, text='Webcam view').pack()
        self.webcam_label = tk.Label(left_frame)
        self.webcam_label.pack()

        tk.Button(left_frame, text='Start Webcam', command=self.start_webcam).pack(side=tk.LEFT, padx=3, pady=5)
        tk.Button(left_frame, text='Stop Webcam', command=self.stop_webcam).pack(side=tk.LEFT, padx=3, pady=5)

        # right side (image select) 
        right_frame = tk.Frame(self.root)
        right_frame.grid(row=0, column=1, padx=10, pady=10)

        tk.Label(right_frame, text='Selected image / annotated').pack()
        self.image_label = tk.Label(right_frame)
        self.image_label.pack()

        tk.Button(right_frame, text='Choose Image', command=self.choose_image).pack(side=tk.LEFT, padx=3, pady=5)
        tk.Button(right_frame, text='Clear', command=self.clear_image).pack(side=tk.LEFT, padx=3, pady=5)

        # bottom: status
        self.status = tk.Label(self.root, text='Trạng thái: sẵn sàng', fg='blue')
        self.status.grid(row=1, column=0, columnspan=2, pady=5)

    def set_status(self, txt):
        self.status.config(text=f'Trạng thái: {txt}')

    def start_webcam(self):
        if self.webcam_running:
            return
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            messagebox.showerror('Lỗi', 'Không mở được webcam.')
            return
        self.webcam_running = True
        self.set_status('Webcam đang chạy')
        threading.Thread(target=self._webcam_loop, daemon=True).start()

    def stop_webcam(self):
        self.webcam_running = False
        if self.cap:
            self.cap.release()
        self.set_status('Webcam dừng')

    def _webcam_loop(self):
        while self.webcam_running and self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                continue
            annotated, alerts = self.engine.process_frame(frame, self.logger)
            self.current_frame = annotated
            self._update_webcam_image(annotated, alerts)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                self.stop_webcam()
                break

    def _update_webcam_image(self, frame, alerts):
        # convert BGR to RGB
        display = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        display = cv2.resize(display, (640, 360))
        imgtk = ImageTk.PhotoImage(Image.fromarray(display))
        self.webcam_label.imgtk = imgtk
        self.webcam_label.configure(image=imgtk)
        self.set_status(f'Webcam on | Alerts: {len(alerts)}')

    def choose_image(self):
        filetypes = [('Image files', '*.jpg *.jpeg *.png *.bmp *.tiff *.tif *.webp'), ('All files', '*.*')]
        path = filedialog.askopenfilename(title='Chọn ảnh', filetypes=filetypes)
        if not path:
            return

        img = cv2.imread(path)
        if img is None:
            messagebox.showerror('Lỗi', f'Không đọc được ảnh: {path}')
            return

        annotated, alerts = self.engine.process_frame(img, self.logger)
        self._display_selected_image(annotated)
        self.set_status(f'Image loaded | Alerts: {len(alerts)}')

    def clear_image(self):
        self.image_label.config(image='')
        self.image_label.imgtk = None
        self.set_status('Đã xoá ảnh chọn')

    def _display_selected_image(self, frame):
        display = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        display = cv2.resize(display, (640, 360))
        imgtk = ImageTk.PhotoImage(Image.fromarray(display))
        self.image_label.imgtk = imgtk
        self.image_label.config(image=imgtk)


def main():
    root = tk.Tk()
    app = NoPlateGUI(root)
    root.protocol('WM_DELETE_WINDOW', lambda: (app.stop_webcam(), root.destroy()))
    root.mainloop()


if __name__ == '__main__':
    main()
