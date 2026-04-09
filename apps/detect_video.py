import cv2
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parent.parent))
from scr.detection.no_plate_engine import NoPlateEngine
from scr.logging.violation_logger import ViolationLogger
import config


def process_video(video_path, output_path=None):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f'Khong mo duoc video: {video_path}')
        return False

    engine = NoPlateEngine(enable_ocr=config.ENABLE_OCR)
    logger = ViolationLogger()

    writer = None
    if output_path:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        fps = max(1.0, cap.get(cv2.CAP_PROP_FPS) or 25.0)
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        writer = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        output, alerts = engine.process_frame(frame, logger)

        for a in alerts:
            print(f'Alert: {a["label"]}, conf={a["conf"]:.2f}, bbox={a["bbox"]}')

        cv2.imshow('No Plate Detection', output)
        if writer:
            writer.write(output)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    if writer:
        writer.release()

    cv2.destroyAllWindows()
    logger.summary()
    return True


def menu_loop():
    from apps.detect_image import process_image

    while True:
        print('\n=== MENU No-Plate Detection ===')
        print('1. Xử lý ảnh tĩnh')
        print('2. Xử lý video tĩnh')
        print('3. Mở webcam realtime')
        print('4. Thoát')

        choice = input('Chọn (1-4): ').strip()

        if choice == '1':
            img_path = input('Đường dẫn ảnh (Enter để dùng default License-Plate-Recognition-3/test/images): ').strip()
            if not img_path:
                img_dir = Path('License-Plate-Recognition-3/test/images')
                imgs = sorted(list(img_dir.glob('*.jpg')) + list(img_dir.glob('*.png')))
                if not imgs:
                    print('Không tìm thấy ảnh default, hãy cung cấp đường dẫn chính xác.')
                    continue
                img_path = str(imgs[0])
                print('Dùng ảnh mặc định:', img_path)
            out = input('Lưu output ảnh (Enter để không lưu): ').strip() or None
            process_image(img_path, save_path=out)

        elif choice == '2':
            vid_path = input('Đường dẫn video (Enter để dùng default hoặc 0 để dang webcam): ').strip()
            if not vid_path:
                vid_path = input('Nhập đường dẫn video trong thư mục hiện tại: ').strip()
                if not vid_path:
                    print('Chưa cung cấp video. Quay lại menu.')
                    continue
            out_path = input('Lưu output video (Enter để không lưu): ').strip() or None
            process_video(vid_path, output_path=out_path)

        elif choice == '3':
            out_path = input('Lưu output webcam video (Enter để không lưu): ').strip() or None
            process_video(0, output_path=out_path)

        elif choice == '4':
            print('Thoát.')
            break

        else:
            print('Lựa chọn không hợp lệ, vui lòng thử lại.')


if __name__ == '__main__':
    menu_loop()

