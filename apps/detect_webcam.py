import cv2
import sys
from pathlib import Path
from datetime import datetime

# Đảm bảo folder gốc chứa package `scr` trong sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from scr.detection.no_plate_engine import NoPlateEngine
from scr.logging.violation_logger import ViolationLogger
import config

def process_frame(frame, engine, logger):
    annotated, alerts = engine.process_frame(frame, logger)
    status_text = f"Alerts: {len(alerts)} - q:quit s:screenshot i:image v:video m:webcam"
    cv2.putText(annotated, status_text, (10, 30), config.FONT, 0.7, (255, 255, 255), 2)
    return annotated, alerts


def run_app():
    engine = NoPlateEngine(enable_ocr=True)
    logger = ViolationLogger()
    mode = 'webcam'
    cap = None

    def open_webcam():
        nonlocal cap
        if cap is not None:
            cap.release()
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    open_webcam()
    print("[INFO] Chương trình bắt đầu. Nhấn 'i' để ảnh, 'v' để video, 'm' để webcam, 'q' để thoát.")

    output_writer = None
    while True:
        if mode == 'webcam':
            ret, frame = cap.read()
            if not ret:
                print("Webcam frame not available")
                break
            annotated, alerts = process_frame(frame, engine, logger)

            if output_writer is not None:
                output_writer.write(annotated)

        elif mode == 'image':
            img_path = input('Nhập đường dẫn ảnh test (Enter để dùng mặc định): ').strip()
            if not img_path:
                img_path = str(Path('License-Plate-Recognition-3/test/images').glob('*.jpg').__iter__().__next__())
            frame = cv2.imread(img_path)
            if frame is None:
                print('Không đọc được ảnh. Về webcam.')
                mode = 'webcam'
                continue
            annotated, alerts = process_frame(frame, engine, logger)
            if output_writer is not None:
                output_writer.write(annotated)

        elif mode == 'video':
            vid_path = input('Nhập đường dẫn video test: ').strip()
            cap_video = cv2.VideoCapture(vid_path)
            if not cap_video.isOpened():
                print('Không mở được video, về webcam.')
                mode = 'webcam'
                continue
            fps = cap_video.get(cv2.CAP_PROP_FPS) or 25
            w = int(cap_video.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap_video.get(cv2.CAP_PROP_FRAME_HEIGHT))
            if output_writer is not None:
                output_writer.release(); output_writer = None
            while True:
                ret, frame = cap_video.read()
                if not ret:
                    break
                annotated, alerts = process_frame(frame, engine, logger)
                cv2.imshow('No Plate Detector + OCR', annotated)
                if cv2.waitKey(int(1000/fps)) & 0xFF == ord('q'):
                    break
            cap_video.release()
            mode = 'webcam'
            continue

        cv2.imshow('No Plate Detector + OCR', annotated)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break
        elif key == ord('s'):
            ts = datetime.now().strftime('%Y%m%d_%H%M%S')
            cv2.imwrite(f'screenshot_{ts}.jpg', annotated)
            print('Screenshot saved.')
        elif key == ord('i'):
            mode = 'image'
        elif key == ord('v'):
            mode = 'video'
        elif key == ord('m'):
            mode = 'webcam'
            open_webcam()

    if cap is not None:
        cap.release()
    if output_writer is not None:
        output_writer.release()
    cv2.destroyAllWindows()
    logger.summary()

if __name__ == "__main__":
    run_app()