from pathlib import Path
import sys
from roboflow import Roboflow
import os

sys.path.append(str(Path(__file__).resolve().parent.parent))
import config


def download_plate_dataset(api_key=None, workspace=None, project_name=None, version=3, fmt='yolov8'):
    api_key = api_key or config.ROBOFLOW_API_KEY
    if not api_key or api_key.lower().startswith('your_key'):
        raise ValueError('Không tìm thấy ROBOFLOW_API_KEY. Hãy đặt giá trị trong config.py hoặc biến môi trường.')

    workspace = workspace or 'ls-workspace-pdyii'
    project_name = project_name or 'license-plate-recognition-rxg4e-urbwk'

    print(f"Roboflow: workspace={workspace}, project={project_name}, version={version}, format={fmt}")
    rf = Roboflow(api_key=api_key)
    dataset = rf.workspace(workspace).project(project_name).version(version).download(fmt)
    print(f"Dataset downloaded to: {dataset.location}")
    print('Các file dataset:')
    for f in os.listdir(dataset.location):
        print(' -', f)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Download dataset từ Roboflow')
    parser.add_argument('--api-key', default=None, help='Roboflow API key')
    parser.add_argument('--workspace', default='ls-workspace-pdyii', help='Roboflow workspace ID (example: ls-workspace-pdyii)')
    parser.add_argument('--project', default='license-plate-recognition-rxg4e-urbwk', help='Tên project')
    parser.add_argument('--version', type=int, default=3, help='Phiên bản dataset')
    parser.add_argument('--format', default='yolov8', help='Định dạng tải về (yolov8, darknet, etc)')
    args = parser.parse_args()

    download_plate_dataset(
        api_key=args.api_key,
        workspace=args.workspace,
        project_name=args.project,
        version=args.version,
        fmt=args.format
    )
if __name__ == "__main__":
    download_plate_dataset()