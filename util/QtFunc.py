import os
from PIL import Image
import xml.etree.ElementTree as ET

from PyQt5.QtWidgets import QMainWindow, QApplication, QMessageBox


def upWindowsh(hint):
    messBox = QMessageBox()
    messBox.setWindowTitle(u'提示')
    messBox.setText(hint)
    messBox.exec_()


def list_images_in_directory(directory):
    image_files = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):
                image_files.append(os.path.join(root, file))
    return image_files


# 修改照片大小
def Change_image_Size(image_path):
    """
    生成用于显示的缩放图，但**不覆盖写入原始图片文件**。

    之前的实现会把缩放后的图片直接保存回 image_path，导致：
    - 图片实际分辨率被永久改写（例如 3200x1800 变成 1300x731）
    - 其它标注软件再打开同名图片时，看到的是被改过尺寸的图片
    - 从而出现“坐标变了/框错位”的现象
    """
    original_image = Image.open(image_path)
    orig_w, orig_h = original_image.size

    # 只缩小不放大：避免小图被强行拉伸
    ratio = min(1300 / orig_w, 850 / orig_h, 1.0)
    disp_w = int(orig_w * ratio)
    disp_h = int(orig_h * ratio)

    reduced_image = original_image.resize((disp_w, disp_h))

    # 写入缓存目录，避免污染原图
    cache_dir = os.path.join("GUI", ".cache_images")
    os.makedirs(cache_dir, exist_ok=True)
    base = os.path.basename(image_path)
    name, ext = os.path.splitext(base)
    cache_path = os.path.join(cache_dir, f"{name}_{disp_w}x{disp_h}{ext if ext else '.jpg'}")

    try:
        reduced_image.save(cache_path)
    except Exception:
        cache_path = os.path.join(cache_dir, f"{name}_{disp_w}x{disp_h}.png")
        reduced_image.save(cache_path)

    # 返回：显示用图片路径、显示尺寸、原图尺寸
    return cache_path, int(disp_w), int(disp_h), int(orig_w), int(orig_h)


def list_label(label_path):
    with open(label_path, 'r') as file:
        content = file.read()
    root = ET.fromstring(content)
    objects = root.findall('object')

    list_labels = []
    list_box = []
    for obj in objects:
        name = obj.find('name').text
        xmin = int(obj.find('bndbox/xmin').text)
        ymin = int(obj.find('bndbox/ymin').text)
        xmax = int(obj.find('bndbox/xmax').text)
        ymax = int(obj.find('bndbox/ymax').text)

        box = [xmin, ymin, xmax, ymax]
        list_labels.append(name)
        list_box.append(box)
    return list_labels, list_box


def get_labels(label_path):
    with open(label_path, 'r') as file:
        content = file.read()
    root = ET.fromstring(content)

    get_list_label = []

    for obj in root.findall('object'):
        xmin = int(obj.find('bndbox/xmin').text)
        ymin = int(obj.find('bndbox/ymin').text)
        xmax = int(obj.find('bndbox/xmax').text)
        ymax = int(obj.find('bndbox/ymax').text)

        item = {
            'name': obj.find('name').text,
            'pose': obj.find('pose').text,
            'truncated': int(obj.find('truncated').text),
            'difficult': int(obj.find('difficult').text),
            'bndbox': [xmin, ymin, xmax, ymax],
        }
        get_list_label.append(item)

    return get_list_label


