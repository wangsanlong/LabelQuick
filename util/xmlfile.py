import os
import xml.etree.ElementTree as ET


# 排版
def indent(elem, level=0):
    i = "\n" + level*"\t"
    if len(elem):
        if not elem.text or not elem.text.strip():
            elem.text = i + "\t"
        if not elem.tail or not elem.tail.strip():
            elem.tail = i
        for elem in elem:
            indent(elem, level+1)
        if not elem.tail or not elem.tail.strip():
            elem.tail = i
    else:
        if level and (not elem.tail or not elem.tail.strip()):
            elem.tail = i


def xml(image_path, save_path, size, labels):
    """
    将 labels 中的标注按 Pascal VOC 格式写入 XML。
    约定：labels 里的 bndbox 始终为 [xmin, ymin, xmax, ymax] 绝对坐标。
    """
    root = ET.Element('annotation')
    folder_name = os.path.dirname(image_path)
    folder = ET.SubElement(root, 'folder')
    folder.text = folder_name

    file_name = os.path.basename(image_path)
    filename = ET.SubElement(root, 'filename')
    filename.text = file_name

    filepath = ET.SubElement(root, 'path')
    filepath.text = image_path

    img_size = ET.SubElement(root, 'size')
    width = ET.SubElement(img_size, 'width')
    width.text = str(size[0])
    height = ET.SubElement(img_size, 'height')
    height.text = str(size[1])
    depth = ET.SubElement(img_size, 'depth')
    depth.text = str(size[2])

    for dic in labels:
        obj = ET.SubElement(root, 'object')

        lab_name = ET.SubElement(obj, 'name')
        lab_name.text = dic['name']

        pose = ET.SubElement(obj, 'pose')
        pose.text = dic['pose']

        truncated = ET.SubElement(obj, 'truncated')
        truncated.text = str(dic['truncated'])

        difficult = ET.SubElement(obj, 'difficult')
        difficult.text = str(dic['difficult'])

        xmin_val = int(dic['bndbox'][0])
        ymin_val = int(dic['bndbox'][1])
        xmax_val = int(dic['bndbox'][2])
        ymax_val = int(dic['bndbox'][3])

        bndbox = ET.SubElement(obj, 'bndbox')
        xmin = ET.SubElement(bndbox, 'xmin')
        xmin.text = str(xmin_val)
        ymin = ET.SubElement(bndbox, 'ymin')
        ymin.text = str(ymin_val)
        xmax = ET.SubElement(bndbox, 'xmax')
        xmax.text = str(xmax_val)
        ymax = ET.SubElement(bndbox, 'ymax')
        ymax.text = str(ymax_val)

    indent(root)  # 格式化xml
    tree = ET.ElementTree(root)
    tree.write(save_path)  # 写入文件
    return tree


def xml_message(save_path, image_name, img_width, img_height, text, x, y, w, h):
    """
    从左上角坐标 (x, y) 和宽高 (w, h) 生成一条标注记录，
    并将其转换为 Pascal VOC 绝对坐标 [xmin, ymin, xmax, ymax] 存到 bndbox。
    """
    file_path = os.path.join(save_path, f"{image_name}.xml")
    size = [img_width, img_height, 3]
    xmin = int(x)
    ymin = int(y)
    xmax = int(x + w)
    ymax = int(y + h)
    result = {
        'name': text,
        'pose': 'Unspecified',
        'truncated': 0,
        'difficult': 0,
        'bndbox': [xmin, ymin, xmax, ymax],
    }
    return result, file_path, size


if __name__ == "__main__":
    path = r'dog.4019.jpg'
    save_path = "111.xml"
    size = [640, 640, 3]
    labels = [{'name': 'body',
               'pose': 'Unspecified',
               'truncated': 0,
               'difficult': 0,
               'bndbox': [9, 89, 297, 305]},
              {'name': 'body',
               'pose': 'Unspecified',
               'truncated': 0,
               'difficult': 1,
               'bndbox': [20, 89, 297, 350]}]

    print(xml(path,save_path, size, labels))
