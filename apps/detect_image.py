import cv2
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parent.parent))
from scr.detection.no_plate_engine import NoPlateEngine
from scr.logging.violation_logger import ViolationLogger
import config


def process_image(image_path, save_path=None):
    img = cv2.imread(image_path)
    if img is None:
        print(f'Khong doc duoc anh: {image_path}')
        return False

    engine = NoPlateEngine(enable_ocr=config.ENABLE_OCR)
    logger = ViolationLogger()
    output, alerts = engine.process_frame(img, logger)

    for i, a in enumerate(alerts, 1):
        print(f'Alert {i}: {a["label"]}, conf={a["conf"]:.2f}, bbox={a["bbox"]}')

    cv2.imshow('No Plate Detection', output)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    if save_path:
        cv2.imwrite(save_path, output)
        print('Saved annotated image to', save_path)

    return True

    for i, a in enumerate(alerts, 1):
        print(f'Alert {i}: {a["label"]}, conf={a["conf"]:.2f}, bbox={a["bbox"]}')

    cv2.imshow('No Plate Detection', output)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    import argparse
    from glob import glob

    parser = argparse.ArgumentParser(description='Detect no-plate vehicles in image (tĩnh)')
    parser.add_argument('--image', help='Path to input image file')
    parser.add_argument('--dir', help='Folder chứa ảnh test')
    parser.add_argument('--output', help='Lưu ảnh đã annotation', default=None)
    args = parser.parse_args()

    img_paths = []
    if args.image:
        img_paths = [args.image]
    elif args.dir:
        patterns = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff', '*.tif', '*.webp']
        img_paths = []
        for p in patterns:
            img_paths += sorted(glob(str(Path(args.dir) / p)))
    else:
        default_dir = Path('License-Plate-Recognition-3/test/images')
        if default_dir.exists():
            patterns = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff', '*.tif', '*.webp']
            img_paths = []
            for p in patterns:
                img_paths += sorted(default_dir.glob(p))
        else:
            print('Cần cung cấp --image hoặc --dir, hoặc đặt ảnh trong License-Plate-Recognition-3/test/images')
            sys.exit(1)

    if not img_paths:
        print('Không tìm thấy ảnh để xử lý.')
        sys.exit(1)

    for idx, p in enumerate(img_paths, 1):
        out_path = None
        if args.output:
            out_path = str(Path(args.output).with_suffix(f'.{idx}.jpg')) if Path(args.output).is_dir() else args.output
        process_image(str(p), save_path=out_path)

