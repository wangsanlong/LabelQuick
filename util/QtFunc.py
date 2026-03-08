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
    # 打开原图像
    original_image = Image.open(image_path)
    # 获取照片大小
    width, height = original_image.size
    ratio = 1300 / width
    width = 1300
    height *= ratio
    reduced_image = original_image.resize((int(width), int(height)))
    if height > 850:
        ratio = 850 / height
        height = 850
        width *= ratio
        reduced_image = original_image.resize((int(width), int(height)))
    reduced_image.save((image_path))
    return image_path, int(width), int(height)


def _get_image_size_from_xml(root, label_path):
    """
    根据 XML 中的 <path>/<folder>/<filename> 尝试找到实际图像，
    返回该图像的 (width, height)。找不到则返回 (None, None)。
    """
    img_path = None
    path_node = root.find("path")
    if path_node is not None and path_node.text:
        candidate = path_node.text
        if os.path.exists(candidate):
            img_path = candidate

    if img_path is None:
        folder_node = root.find("folder")
        filename_node = root.find("filename")
        if folder_node is not None and filename_node is not None:
            candidate = os.path.join(folder_node.text, filename_node.text)
            if os.path.exists(candidate):
                img_path = candidate

    if img_path is None:
        filename_node = root.find("filename")
        if filename_node is not None:
            candidate = os.path.join(os.path.dirname(label_path), filename_node.text)
            if os.path.exists(candidate):
                img_path = candidate

    if img_path is None:
        return None, None

    try:
        with Image.open(img_path) as im:
            return im.size
    except Exception:
        return None, None


def _get_scale_from_xml(label_path):
    """
    基于 XML 记录的 size 和当前图像真实尺寸，计算坐标缩放比例。
    当 YOLO 在原始 3200x1800 上标注，而本项目把图像缩小到 1300xH 时，
    会用该比例把标注同步缩放，保证坐标和当前图像对齐。
    """
    with open(label_path, "r") as file:
        content = file.read()
    root = ET.fromstring(content)

    xml_w = xml_h = None
    size_node = root.find("size")
    if size_node is not None:
        w_node = size_node.find("width")
        h_node = size_node.find("height")
        if w_node is not None and h_node is not None:
            try:
                xml_w = int(w_node.text)
                xml_h = int(h_node.text)
            except (TypeError, ValueError):
                xml_w = xml_h = None

    img_w, img_h = _get_image_size_from_xml(root, label_path)

    if not xml_w or not xml_h or not img_w or not img_h:
        return 1.0, 1.0, root

    scale_x = img_w / float(xml_w)
    scale_y = img_h / float(xml_h)
    return scale_x, scale_y, root


def list_label(label_path):
    # 计算缩放比例并解析 XML
    scale_x, scale_y, root = _get_scale_from_xml(label_path)

    objects = root.findall('object')

    list_labels = []
    list_box = []
    for obj in objects:
        name = obj.find('name').text
        xmin = int(obj.find('bndbox/xmin').text)
        ymin = int(obj.find('bndbox/ymin').text)
        xmax = int(obj.find('bndbox/xmax').text)
        ymax = int(obj.find('bndbox/ymax').text)

        # 把基于原始分辨率的坐标缩放到当前图像分辨率
        xmin = int(round(xmin * scale_x))
        ymin = int(round(ymin * scale_y))
        xmax = int(round(xmax * scale_x))
        ymax = int(round(ymax * scale_y))

        box = [xmin, ymin, xmax, ymax]
        list_labels.append(name)
        list_box.append(box)
    return list_labels, list_box


def get_labels(label_path):
    # 同 list_label，一样做比例缩放，保证 bndbox 始终和当前图像尺寸一致
    scale_x, scale_y, root = _get_scale_from_xml(label_path)

    get_list_label = []

    for obj in root.findall('object'):
        xmin = int(obj.find('bndbox/xmin').text)
        ymin = int(obj.find('bndbox/ymin').text)
        xmax = int(obj.find('bndbox/xmax').text)
        ymax = int(obj.find('bndbox/ymax').text)

        xmin = int(round(xmin * scale_x))
        ymin = int(round(ymin * scale_y))
        xmax = int(round(xmax * scale_x))
        ymax = int(round(ymax * scale_y))

        item = {
            'name': obj.find('name').text,
            'pose': obj.find('pose').text,
            'truncated': int(obj.find('truncated').text),
            'difficult': int(obj.find('difficult').text),
            'bndbox': [xmin, ymin, xmax, ymax],
        }
        get_list_label.append(item)

    return get_list_label


