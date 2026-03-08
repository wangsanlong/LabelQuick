import sys, os
import copy
import xml.etree.ElementTree as ET
from PyQt5 import QtGui, QtWidgets
from PyQt5.QtGui import QImage, QPixmap, QPainter, QPen
from PyQt5.QtCore import Qt, QCoreApplication, QRect, pyqtSignal, QTimer, QEvent, QLineF, QUrl
from PyQt5.QtMultimedia import QMediaPlayer, QMediaContent
import cv2
import numpy as np
from util.QtFunc import *
from util.xmlfile import *

from GUI.UI_Main import Ui_MainWindow
from GUI.message import LabelInputDialog
from GUI.tag_management import TagManagementDialog

sys.path.append("smapro")
from sampro.LabelQuick_TW import Anything_TW
from sampro.LabelVideo_TW import AnythingVideo_TW

from PyQt5.QtCore import QThread, pyqtSignal, QTimer

class VideoProcessingThread(QThread):
    finished = pyqtSignal()  # 完成信号
    frame_ready = pyqtSignal(object)  # 添加新信号用于传递处理后的帧

    def __init__(self, avt, video_path, output_dir,clicked_x, clicked_y, method,text,save_path):
        super().__init__()
        self.AVT = AnythingVideo_TW()
        self.video_path = video_path
        self.output_dir = output_dir
        self.clicked_x = clicked_x
        self.clicked_y = clicked_y
        self.method = method
        self.text = text
        self.save_path = save_path
        self.xml_messages = []
        os.makedirs(self.output_dir, exist_ok=True)

    def run(self):
        print(self.clicked_x, self.clicked_y, self.method)
        try:
            # 创建输出目录和mask子目录
            os.makedirs(self.output_dir, exist_ok=True)
            mask_dir = os.path.join(self.output_dir, "mask")
            os.makedirs(mask_dir, exist_ok=True)
            
            # 提取视频帧
            self.AVT.extract_frames_from_video(self.video_path, self.output_dir,fps=2)
            self.AVT.set_video(self.output_dir)
            self.AVT.inference(self.output_dir)
            self.AVT.Set_Clicked([self.clicked_x, self.clicked_y], self.method)
            self.AVT.add_new_points_or_box()
            
            # 获取处理后的帧并发送信号
            processed_frame, xml_messages = self.AVT.Draw_Mask_at_frame(save_image_path=mask_dir,save_path=self.save_path,text=self.text)  # 使用新的mask_dir路径
            self.xml_messages = xml_messages
            self.frame_ready.emit(processed_frame)  # 发送处理后的帧
            
        except Exception as e:
            print(f"处理出错: {str(e)}")
            import traceback
            traceback.print_exc()
        self.finished.emit()


class MainFunc(QMainWindow):
    my_signal = pyqtSignal()

    def __init__(self):
        super(MainFunc, self).__init__()
        # 连接应用程序的 aboutToQuit 信号到自定义的槽函数
        QCoreApplication.instance().aboutToQuit.connect(self.clean_up)
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)

        # 让 scrollArea 根据内部控件真实尺寸决定是否出现滚动条
        self.ui.scrollArea.setWidgetResizable(False)
        self.ui.scrollAreaWidgetContents.setMinimumSize(0, 0)

        # 缩放相关
        self.zoom_factor = 1.0
        # 监听 scrollArea 里的鼠标滚轮事件，用于 Ctrl+滚轮缩放
        self.ui.scrollArea.viewport().installEventFilter(self)

        self.sld_video_pressed=False


        self.image_files = None
        self.img_path = None
        self.save_path = None
        self.clicked_event = False
        self.paint_event = False
        self.labels = []
        self.clicked_save = []
        self.paint_save = []
        self.flag = False
        self.save = True
        self.cap = None
        self.video_path = None

        # 当前图片原始尺寸（经过 Change_image_Size 之后的基准尺寸）
        self.img_width = None
        self.img_height = None

        # 手动矩形编辑 / 四点标注状态
        self.selected_rect_index = None
        self.dragging_rect = False
        self.resizing_rect = False
        self.drag_start_disp = None
        self.drag_start_rect = None  # 基准坐标 [x0,y0,x1,y1]
        self.resize_anchor = None  # 'tl','tr','bl','br','inside'
        self.four_point_mode = False
        self.four_points = []
        self.pending_four_point_rect = None  # 四点待确认框，按 S 后才写入 paint_save
        # 标注模式：默认 sam。可选：'sam' | 'rect' | 'four'
        self.annotate_mode = "sam"
        # 撤回栈：仅对当前图像，每步保存 (image_name, labels, clicked_save, paint_save)
        self.undo_stack = []

        self.AT = Anything_TW()
        self.AVT = AnythingVideo_TW()

        self.timer_camera = QTimer()

        self.ui.actionOpen_Dir.triggered.connect(self.get_dir)
        self.ui.actionNext_Image.triggered.connect(self.next_img)
        self.ui.actionPrev_Image.triggered.connect(self.prev_img)
        self.ui.actionChange_Save_Dir.triggered.connect(self.set_save_path)
        # 标注方式切换
        self.ui.actionCreate_RectBox.triggered.connect(self.toggle_rect_mode)
        self.ui.actionOpen_Video.triggered.connect(self.get_video)
        self.ui.actionVideo_marking.triggered.connect(self.video_marking)


        self.ui.pushButton.clicked.connect(self.Btn_Start)
        self.ui.pushButton_2.clicked.connect(self.Btn_Stop)
        self.ui.pushButton_3.clicked.connect(self.Btn_Save)
        self.ui.pushButton_4.clicked.connect(self.Btn_Replay)
        self.ui.pushButton_5.clicked.connect(self.Btn_Auto)
        self.ui.pushButton_start_marking.clicked.connect(self.Btn_Start_Marking)

        self.ui.horizontalSlider.sliderReleased.connect(self.releaseSlider)
        self.ui.horizontalSlider.sliderPressed.connect(self.pressSlider)
        self.ui.horizontalSlider.sliderMoved.connect(self.moveSlider)

        # 获取视频总帧数和当前帧位置
        self.total_frames = 0
        self.current_frame = 0

        # 连接标签管理按钮
        self.ui.pushButton_delete_label.clicked.connect(self.delete_selected_label)
        self.ui.pushButton_clear_labels.clicked.connect(self.clear_all_labels)
        self.ui.pushButton_tag_management.clicked.connect(self.open_tag_management)

        # 连接四点标注动作
        self.ui.actionFour_Point_Box.triggered.connect(self.toggle_four_point_mode)
        self.ui.actionSam.triggered.connect(self.toggle_sam_mode)

        # 初始应用默认模式（SAM）
        self.apply_annotate_mode("sam")

    def apply_annotate_mode(self, mode: str):
        """应用并保持当前标注模式，并同步工具栏勾选状态。"""
        if mode not in ("sam", "rect", "four"):
            mode = "sam"
        self.annotate_mode = mode

        # 工具栏显式显示当前模式（仅一个为选中）
        self.ui.actionSam.setChecked(mode == "sam")
        self.ui.actionCreate_RectBox.setChecked(mode == "rect")
        self.ui.actionFour_Point_Box.setChecked(mode == "four")

        # 退出四点模式时清理点
        if mode != "four":
            self.four_point_mode = False
            self.four_points = []

        if mode == "sam":
            self.four_point_mode = False
            self.ui.label_4.mousePressEvent = self.mouse_press_event
            self.ui.label_4.mouseMoveEvent = None
            self.ui.label_4.mouseReleaseEvent = None
            # label_4 的 paintEvent 交给默认 QWidget
            self.ui.label_4.setCursor(Qt.ArrowCursor)
        elif mode == "rect":
            self.four_point_mode = False
            self.paint_event = True
            self.clicked_event = False
            self.ui.label_4.mousePressEvent = self.mousePressEvent
            self.ui.label_4.mouseMoveEvent = self.mouseMoveEvent
            self.ui.label_4.mouseReleaseEvent = self.mouseReleaseEvent
            self.ui.label_4.paintEvent = self.paintEvent
            self.ui.label_4.setCursor(Qt.CrossCursor)
            # 允许持续画框/编辑，不自动退出
        elif mode == "four":
            # 进入四点模式
            self.start_four_point_mode()

    def toggle_rect_mode(self):
        """点击一次进入/切换到矩形模式；再点一次退出回 SAM。"""
        if self.annotate_mode == "rect":
            self.apply_annotate_mode("sam")
        else:
            self.apply_annotate_mode("rect")

    def toggle_sam_mode(self):
        """点击 SAM 则切换到 SAM 分割模式。"""
        self.apply_annotate_mode("sam")

    def toggle_four_point_mode(self):
        """点击一次进入/切换到四点模式；再点一次退出回 SAM。"""
        if self.annotate_mode == "four":
            self.apply_annotate_mode("sam")
        else:
            self.apply_annotate_mode("four")

    # 事件过滤器：用于 Ctrl + 鼠标滚轮缩放
    def eventFilter(self, source, event):
        if source is self.ui.scrollArea.viewport() and event.type() == QEvent.Wheel:
            # 仅在有图像且按下 Ctrl 时启用缩放
            if self.img_path and (event.modifiers() & Qt.ControlModifier):
                delta = event.angleDelta().y()
                if delta == 0:
                    return True
                # 缩放步长
                if delta > 0:
                    self.zoom_factor *= 1.1
                else:
                    self.zoom_factor /= 1.1
                # 限制缩放范围
                self.zoom_factor = max(0.2, min(5.0, self.zoom_factor))

                # 重新按当前缩放比例显示图像，由 QScrollArea 提供滚动条用于拖动查看完整图像
                if hasattr(self, "current_cv_image"):
                    self.update_display_with_image(self.current_cv_image)
                else:
                    self.show_qt(self.img_path)
                return True
        return super(MainFunc, self).eventFilter(source, event)

    def update_display_with_pixmap(self, pixmap):
        """
        根据当前 zoom_factor 将 QPixmap 显示到 label_3 上，
        并同步调整 label_4 的尺寸，保证交互区域一致。
        """
        if not pixmap:
            return
        if self.img_width is None or self.img_height is None:
            w = pixmap.width()
            h = pixmap.height()
        else:
            w = self.img_width
            h = self.img_height

        scaled_w = int(w * self.zoom_factor)
        scaled_h = int(h * self.zoom_factor)
        if scaled_w <= 0 or scaled_h <= 0:
            return

        scaled_pix = pixmap.scaled(scaled_w, scaled_h, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        disp_w = scaled_pix.width()
        disp_h = scaled_pix.height()
        self.ui.label_3.setFixedSize(disp_w, disp_h)
        self.ui.label_3.setPixmap(scaled_pix)
        self.ui.label_4.setFixedSize(disp_w, disp_h)

        # 同步放大 scrollArea 的内容部件，让右/下滚动条生效
        self.ui.scrollAreaWidgetContents.resize(disp_w + 20, disp_h + 20)

    def update_display_with_image(self, cv_image):
        """
        使用 OpenCV 图像作为基准，根据当前 zoom_factor 在界面上展示。
        """
        if cv_image is None:
            return
        # 保存当前显示的 CV 图像，方便缩放时复用
        self.current_cv_image = cv_image
        image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
        h, w, channels = image.shape
        bytes_per_line = channels * w
        q_image = QImage(image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        pixmap = QtGui.QPixmap.fromImage(q_image)
        self.update_display_with_pixmap(pixmap)

    def Change_Enable(self,method="",state=False):
        if method=="ShowVideo":
            self.ui.pushButton.setEnabled(state)
            self.ui.pushButton_2.setEnabled(state)
            self.ui.pushButton_3.setEnabled(state)
            self.ui.pushButton_4.setEnabled(state)  # 初始时禁用重播按钮
            self.ui.pushButton_5.setEnabled(state)
            self.ui.horizontalSlider.setEnabled(state)
        if method=="MakeTag":
            self.ui.actionPrev_Image.setEnabled(state)
            self.ui.actionNext_Image.setEnabled(state)
            self.ui.actionSam.setEnabled(state)
            self.ui.actionCreate_RectBox.setEnabled(state)
            self.ui.actionFour_Point_Box.setEnabled(state)
            
    def get_dir(self):
        self.ui.listWidget.clear()
        if self.cap:
            self.timer_camera.stop()
            self.ui.listWidget.clear()  # 清空listWidget
        self.directory = QtWidgets.QFileDialog.getExistingDirectory()
        if self.directory:
            self.undo_stack = []
            self.image_files = list_images_in_directory(self.directory)
            self.current_index = 0
            # 每次重新选择目录时重置缩放
            self.zoom_factor = 1.0
            self.show_path_image()
            self.Change_Enable(method="MakeTag",state=True)
            self.Change_Enable(method="ShowVideo",state=False)
            # 禁用开始检测打标按钮
            self.ui.pushButton_start_marking.setEnabled(False)
            # 鼠标点击触发
            # 默认进入 SAM 分割模式并保持
            self.apply_annotate_mode("sam")

    def show_path_image(self):
        if self.image_files:
            self.image_path = self.image_files[self.current_index]
            self.img_path = self.image_path
            self.image_name = os.path.basename(self.image_path).split('.')[0]
            # 更新 UI 显示当前图像名称并在后台打印
            base_name = os.path.basename(self.image_path)
            self.ui.label_image_name.setText(f"当前图像：{base_name}")
            print(f"当前标注图像：{base_name}")

            # 调整图像大小到统一基准尺寸，并记录基准尺寸
            self.img_path, self.img_width, self.img_height = Change_image_Size(self.img_path)
            self.image = cv2.imread(self.img_path)
            self.AT.Set_Image(self.image)
            # 保存当前 CV 图像用于缩放
            self.current_cv_image = self.image.copy()
            self.show_qt(self.img_path)
            self.Exists_Labels_And_Boxs()

    # 展示已保存所有标签
    def Exists_Labels_And_Boxs(self):
        self.list_labels = []
        label_file = os.path.exists(f"{self.save_path}/{self.image_name}.xml")
        if label_file:
            label_path = f"{self.save_path}/{self.image_name}.xml"
            self.labels = get_labels(label_path)
            self.list_labels,list_box = list_label(label_path)
            self.paint_save = list_box
            self.Show_Exists()
            for label in self.list_labels:
                self.ui.listWidget.addItem(label)

    def show_qt(self, img_path):
        if img_path != None:
            Qt_Gui = QtGui.QPixmap(img_path)
            self.update_display_with_pixmap(Qt_Gui)

    def next_img(self):
        if self.img_path and not self.clicked_event and not self.paint_event:
            if self.image_files and self.current_index < len(self.image_files) - 1:
                self.current_index += 1
                print(self.current_index)
                self.Other_Img()
            else:
                upWindowsh("这是最后一张")

    def prev_img(self):
        if self.img_path and not self.clicked_event and not self.paint_event:
            if self.image_files and self.current_index > 0:
                self.current_index -= 1
                self.Other_Img()
                
            else:
                upWindowsh("这是第一张")
                
    def Other_Img(self):
        self.undo_stack = []
        self.labels = []
        self.paint_save = []
        self.clicked_save = []
        self.ui.listWidget.clear()
        # 切换图片时重置缩放
        self.zoom_factor = 1.0
        self.show_path_image()
        # 切换图片后保持当前模式（若当前是 four，则重置四点点位）
        self.apply_annotate_mode(self.annotate_mode)

    def set_save_path(self):
        directory = QtWidgets.QFileDialog.getExistingDirectory()
        if directory:
            self.save_path = directory
            if self.img_path:
                self.Exists_Labels_And_Boxs()

# ########################################################################################################################
    # seg
    def mouse_press_event(self, event):
        try:
            if self.img_path:
                self.clicked_event = True
                # 将当前显示坐标转换为基准图像坐标（考虑任意缩放比例）
                disp_x = event.x()
                disp_y = event.y()
                x = disp_x
                y = disp_y
                if self.img_width and self.img_height:
                    disp_w = self.ui.label_3.width()
                    disp_h = self.ui.label_3.height()
                    if disp_w > 0 and disp_h > 0:
                        scale_x = self.img_width / float(disp_w)
                        scale_y = self.img_height / float(disp_h)
                        x = int(disp_x * scale_x)
                        y = int(disp_y * scale_y)

                if event.button() == Qt.LeftButton:
                    self.clicked_x, self.clicked_y, self.method = x, y, 1
                if event.button() == Qt.RightButton:
                    self.clicked_x, self.clicked_y, self.method = x, y, 0

                image = self.image.copy()
                self.AT.Set_Clicked([x, y], self.method)
                self.AT.Create_Mask()
                image = self.AT.Draw_Mask(self.AT.mask, image)

                # 保存当前显示图像，便于缩放
                self.current_cv_image = image.copy()
                self.update_display_with_image(self.current_cv_image)

                self.save = False
        except Exception as e:
            print(f"Error in mouse_press_event: {str(e)}")

# ########################################################################################################################
# 重写QWidget类的keyPressEvent方法
    def keyPressEvent(self, event):
        if self.img_path:
            # Alt+S：为当前框自动使用第一个标签（优先处理，避免触发普通 S 逻辑）
            if event.key() == Qt.Key_S and (event.modifiers() & Qt.AltModifier):
                if self.ui.listWidget.count() > 0 and (self.clicked_event or self.paint_event):
                    first_item = self.ui.listWidget.item(0)
                    if first_item:
                        text = first_item.text()
                        self.on_dialog_confirmed(text)
                        return
                # 没有可用标签或没有当前框，则交给父类处理
                return super(QMainWindow, self).keyPressEvent(event)

            if self.clicked_event and not self.paint_event:
                image = self.AT.Key_Event(event.key())

            if self.video_path:
                if self.clicked_event or self.paint_event:
                    # 普通 S（无 Alt 等修饰键）才弹出标签输入框
                    if (event.key() == Qt.Key_S) and not (event.modifiers() & (Qt.AltModifier | Qt.ControlModifier | Qt.ShiftModifier)):
                        self.save = True
                        self.dialog = LabelInputDialog(self)
                        self.dialog.show()
                        self.dialog.confirmed.connect(self.video_on_dialog_confirmed)
                        
                        # 禁用label4的鼠标事件
                        self.ui.label_4.mousePressEvent = None

                    if (event.key() == 81):
                        self.clicked_event = False
                        self.paint_event = False
                        self.save = True
                        self.Show_Exists()
                        self.ui.label_4.mousePressEvent = self.mouse_press_event
                        self.ui.label_4.setCursor(Qt.ArrowCursor)
            else:
                if self.clicked_event or self.paint_event:
                    # 普通 S（无 Alt 等修饰键）才弹出标签输入框
                    if (event.key() == Qt.Key_S) and not (event.modifiers() & (Qt.AltModifier | Qt.ControlModifier | Qt.ShiftModifier)):
                        self.save = True
                        self.dialog = LabelInputDialog(self)
                        self.dialog.show()
                        self.dialog.confirmed.connect(self.on_dialog_confirmed)

                    if (event.key() == 81):
                        self.clicked_event = False
                        self.paint_event = False
                        self.save = True
                        self.Show_Exists()
                        self.ui.label_4.mousePressEvent = self.mouse_press_event
                        self.ui.label_4.setCursor(Qt.ArrowCursor)

                

            # Backspace：清空当前图像的所有标签（保持原有逻辑）
            if event.key() == 16777219:
                self.clicked_event = False
                self.paint_event = False
                self.save = True
                self.ui.listWidget.clear()
                self.list_labels = []
                self.clicked_save = []
                self.paint_save = []
                self.show_qt(self.img_path)
                self.ui.label_4.mousePressEvent = self.mouse_press_event
                self.ui.label_4.setCursor(Qt.ArrowCursor)
                if os.path.exists(f"{self.save_path}/{self.image_name}.xml"):
                    os.remove(f"{self.save_path}/{self.image_name}.xml")
                    self.labels = []
                else:
                    super(QMainWindow, self).keyPressEvent(event)

            # Ctrl+Z：撤回上一步
            if event.key() == Qt.Key_Z and (event.modifiers() & Qt.ControlModifier):
                self.undo()
                return

            # ESC：仅取消当前未确认的框，不删除已保存标签
            if event.key() == Qt.Key_Escape:
                if self.clicked_event or self.paint_event or getattr(self, 'pending_four_point_rect', None):
                    self.clicked_event = False
                    self.paint_event = False
                    self.save = True
                    self.pending_four_point_rect = None
                    self.Show_Exists()
                    # 取消后保持当前模式
                    self.apply_annotate_mode(self.annotate_mode)
                else:
                    super(QMainWindow, self).keyPressEvent(event)

            # Alt+S：为当前框自动使用第一个标签
            #（已在函数开头优先处理，这里不再重复）
    def on_dialog_confirmed(self, text):
        if not self.save_path:
            upWindowsh("请选择保存路径")

        elif text and self.clicked_event:
            self.push_undo()
            self.ui.listWidget.addItem(text)
            result, file_path, size = xml_message(self.save_path, self.image_name, self.img_width, self.img_height,
                                                  text, self.AT.x, self.AT.y, self.AT.w, self.AT.h)
            self.labels.append(result)
            self.clicked_save.append([self.AT.x, self.AT.y, (self.AT.w + self.AT.x), (self.AT.h + self.AT.y)])
            xml(self.image_path, file_path, size, self.labels)

        elif text and self.paint_event:
            self.push_undo()
            self.paint_event = False
            self.clicked_event = True
            self.ui.listWidget.addItem(text)

            # 四点标注：直接使用 pending_four_point_rect；否则从显示坐标转换
            if getattr(self, 'pending_four_point_rect', None):
                r = self.pending_four_point_rect
                x0_base, y0_base, x1_base, y1_base = r[0], r[1], r[2], r[3]
                self.pending_four_point_rect = None
            else:
                x0_base = self.x0
                y0_base = self.y0
                x1_base = self.x1
                y1_base = self.y1
                if self.img_width and self.img_height:
                    disp_w = self.ui.label_3.width()
                    disp_h = self.ui.label_3.height()
                    if disp_w > 0 and disp_h > 0:
                        scale_x = self.img_width / float(disp_w)
                        scale_y = self.img_height / float(disp_h)
                        x0_base = int(self.x0 * scale_x)
                        y0_base = int(self.y0 * scale_y)
                        x1_base = int(self.x1 * scale_x)
                        y1_base = int(self.y1 * scale_y)

            result, file_path, size = xml_message(
                self.save_path,
                self.image_name,
                self.img_width,
                self.img_height,
                text,
                x0_base,
                y0_base,
                abs(x1_base - x0_base),
                abs(y1_base - y0_base),
            )
            self.labels.append(result)
            self.paint_save.append([x0_base, y0_base, x1_base, y1_base])
            xml(self.image_path, file_path, size, self.labels)

            # 确认后保持当前模式（不强制切回 SAM）
            self.apply_annotate_mode(self.annotate_mode)
            
        self.clicked_event = False
        self.paint_event = False

        self.Show_Exists()

    def video_on_dialog_confirmed(self, text):
        self.text = text
        print(self.text)
        if not self.save_path:
            upWindowsh("请选择保存路径")
        elif text and self.clicked_event:
            self.ui.listWidget.addItem(text)
            result, file_path, size = xml_message(self.save_path, self.image_name, self.img_width, self.img_height,
                                                    text, self.AT.x, self.AT.y, self.AT.w, self.AT.h)
            self.labels.append(result)
            self.clicked_save.append([self.AT.x, self.AT.y, (self.AT.w + self.AT.x), (self.AT.h + self.AT.y)])
            xml(self.image_path, file_path, size, self.labels)
            # 启用"开始检测打标"按钮
            self.ui.pushButton_start_marking.setEnabled(True)
        self.clicked_event = False
        self.paint_event = False

        self.Show_Exists()
        

    # 显示已存在框
    def Show_Exists(self):
        image = cv2.imread(self.img_path)
        if self.clicked_save == [] and self.paint_save == [] and not getattr(self, 'pending_four_point_rect', None):
            self.show_qt(self.img_path)
        else:
            if self.clicked_save != []:
                for i in self.clicked_save:
                    image = cv2.rectangle(image, (i[0], i[1]), (i[2], i[3]), (0, 255, 0), 2)
            if self.paint_save != []:
                for i in self.paint_save:
                    image = cv2.rectangle(image, (i[0], i[1]), (i[2], i[3]), (0, 0, 255), 2)
            # 绘制四点待确认框（虚线效果用半透明或不同颜色区分）
            if getattr(self, 'pending_four_point_rect', None):
                r = self.pending_four_point_rect
                image = cv2.rectangle(image, (r[0], r[1]), (r[2], r[3]), (255, 165, 0), 2)
            # 保存当前 CV 图像并按缩放比例显示
            self.current_cv_image = image.copy()
            self.update_display_with_image(self.current_cv_image)

# ##################################################################################################
    # 手动打标
    def mousePaint(self):
        if self.img_path != None:
            self.paint_event = True
            self.clicked_event = False
            self.four_point_mode = False
            if self.save:
                self.ui.label_4.mousePressEvent = self.mousePressEvent
                self.ui.label_4.mouseMoveEvent = self.mouseMoveEvent
                self.ui.label_4.mouseReleaseEvent = self.mouseReleaseEvent
                self.ui.label_4.paintEvent = self.paintEvent
                self.ui.label_4.setCursor(Qt.CrossCursor)
                self.save = False
            else:
                upWindowsh("请先输入标签")

    def start_four_point_mode(self):
        """进入四点标注模式：点击四个极值点自动生成矩形框，之后可用拖拽/缩放修改。"""
        if self.img_path is None:
            upWindowsh("请先打开图像")
            return
        # 仅在当前没有未完成的框时允许进入四点模式
        if not self.save:
            upWindowsh("请先为当前框输入或确认标签")
            return
        self.annotate_mode = "four"
        self.four_point_mode = True
        self.four_points = []
        self.paint_event = True
        self.clicked_event = False
        self.ui.label_4.mousePressEvent = self.fourPointMousePressEvent
        self.ui.label_4.mouseMoveEvent = None
        self.ui.label_4.mouseReleaseEvent = None
        self.ui.label_4.paintEvent = self.fourPointPaintEvent
        self.ui.label_4.setCursor(Qt.CrossCursor)

    def fourPointMousePressEvent(self, event):
        """四点标注时的鼠标点击事件：依次记录 4 个点，自动生成矩形框。"""
        if event.button() != Qt.LeftButton:
            return
        # 记录四个极值点（在基准坐标系下）
        disp_x, disp_y = event.pos().x(), event.pos().y()
        x_base, y_base = self._display_to_base(disp_x, disp_y)
        self.four_points.append((x_base, y_base))

            # 达到 4 个点后，根据极值构造矩形（仅设置待确认框，不加入 paint_save，避免与 S 确认时重复添加）
        if len(self.four_points) >= 4:
            xs = [p[0] for p in self.four_points]
            ys = [p[1] for p in self.four_points]
            x_min, x_max = min(xs), max(xs)
            y_min, y_max = min(ys), max(ys)
            # 仅设置待确认框坐标，供 S 键确认时写入；不在此处 append 到 paint_save
            self.pending_four_point_rect = [x_min, y_min, x_max, y_max]
            self.x0, self.y0 = self._base_to_display(x_min, y_min)
            self.x1, self.y1 = self._base_to_display(x_max, y_max)
            self.save = False
            self.paint_event = True  # 使 S / Alt+S 能触发确认
            # 立刻显示矩形（待确认）
            self.Show_Exists()
            # 退出四点模式，切回手动画框交互（可拖拽修改）
            self.four_point_mode = False
            self.four_points = []
            self.ui.label_4.mousePressEvent = self.mousePressEvent
            self.ui.label_4.mouseMoveEvent = self.mouseMoveEvent
            self.ui.label_4.mouseReleaseEvent = self.mouseReleaseEvent
            self.ui.label_4.paintEvent = self.paintEvent

        self.ui.label_4.update()

    def fourPointPaintEvent(self, event):
        """四点模式下，在 label_4 上预览已点击的点。"""
        super(MainFunc, self).paintEvent(event)
        if not self.four_points:
            return
        painter = QPainter(self.ui.label_4)
        painter.setPen(QPen(Qt.red, 2, Qt.DotLine))
        for x_base, y_base in self.four_points:
            x_disp, y_disp = self._base_to_display(x_base, y_base)
            painter.drawEllipse(x_disp - 3, y_disp - 3, 6, 6)

    def _get_scale_factors(self):
        """返回从显示坐标到基准图像坐标的缩放因子 (scale_x, scale_y)。"""
        if not self.img_width or not self.img_height:
            return 1.0, 1.0
        disp_w = self.ui.label_3.width()
        disp_h = self.ui.label_3.height()
        if disp_w <= 0 or disp_h <= 0:
            return 1.0, 1.0
        scale_x = self.img_width / float(disp_w)
        scale_y = self.img_height / float(disp_h)
        return scale_x, scale_y

    def _display_to_base(self, x, y):
        sx, sy = self._get_scale_factors()
        return int(x * sx), int(y * sy)

    def _base_to_display(self, x, y):
        sx, sy = self._get_scale_factors()
        # 反向：基准 -> 显示
        if sx == 0 or sy == 0:
            return x, y
        return int(x / sx), int(y / sy)

    def _hit_test_rect(self, disp_x, disp_y, tolerance=6):
        """
        在显示坐标系下命中测试已有矩形。
        返回 (index, role)，role in {'inside','edge','corner'}。
        """
        if not self.paint_save:
            return None, None
        for idx, box in enumerate(self.paint_save):
            x0, y0 = self._base_to_display(box[0], box[1])
            x1, y1 = self._base_to_display(box[2], box[3])
            left, right = min(x0, x1), max(x0, x1)
            top, bottom = min(y0, y1), max(y0, y1)
            if left - tolerance <= disp_x <= right + tolerance and top - tolerance <= disp_y <= bottom + tolerance:
                # 判断是否靠近边缘用于缩放
                near_left = abs(disp_x - left) <= tolerance
                near_right = abs(disp_x - right) <= tolerance
                near_top = abs(disp_y - top) <= tolerance
                near_bottom = abs(disp_y - bottom) <= tolerance
                if (near_left or near_right) or (near_top or near_bottom):
                    return idx, "edge"
                return idx, "inside"
        return None, None

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            disp_x, disp_y = event.pos().x(), event.pos().y()
            # 先检测是否点在已有矩形上，用于拖拽/缩放
            idx, role = self._hit_test_rect(disp_x, disp_y)
            if idx is not None:
                self.selected_rect_index = idx
                self.drag_start_disp = (disp_x, disp_y)
                self.drag_start_rect = self.paint_save[idx][:]
                if role == "inside":
                    self.dragging_rect = True
                    self.resizing_rect = False
                else:
                    self.dragging_rect = False
                    self.resizing_rect = True
                self.flag = False
            else:
                # 新建矩形
                self.dragging_rect = False
                self.resizing_rect = False
                self.selected_rect_index = None
                self.flag = True
                self.show_qt(self.img_path)
                self.x0, self.y0 = disp_x, disp_y
                self.x1, self.y1 = self.x0, self.y0
            self.ui.label_4.update()

    def mouseReleaseEvent(self, event):
        if self.flag:
            # 完成新建矩形，由 S 键 + 对话框来真正写入 XML
            self.flag = False
        if self.dragging_rect or self.resizing_rect:
            # 拖拽/缩放结束，刷新显示
            self.dragging_rect = False
            self.resizing_rect = False
            self.drag_start_disp = None
            self.drag_start_rect = None
            self.Show_Exists()

    def mouseMoveEvent(self, event):
        disp_x, disp_y = event.pos().x(), event.pos().y()
        if self.flag:
            # 正在新建矩形
            self.x1, self.y1 = disp_x, disp_y
            self.ui.label_4.update()
        elif self.dragging_rect and self.selected_rect_index is not None and self.drag_start_rect is not None:
            # 平移已有矩形（在基准坐标下平移）
            dx_disp = disp_x - self.drag_start_disp[0]
            dy_disp = disp_y - self.drag_start_disp[1]
            sx, sy = self._get_scale_factors()
            dx_base = int(dx_disp * sx)
            dy_base = int(dy_disp * sy)
            x0, y0, x1, y1 = self.drag_start_rect
            new_rect = [x0 + dx_base, y0 + dy_base, x1 + dx_base, y1 + dy_base]
            self.paint_save[self.selected_rect_index] = new_rect
            self.Show_Exists()
        elif self.resizing_rect and self.selected_rect_index is not None and self.drag_start_rect is not None:
            # 简单缩放：只根据鼠标位移调整右下角
            dx_disp = disp_x - self.drag_start_disp[0]
            dy_disp = disp_y - self.drag_start_disp[1]
            sx, sy = self._get_scale_factors()
            dx_base = int(dx_disp * sx)
            dy_base = int(dy_disp * sy)
            x0, y0, x1, y1 = self.drag_start_rect
            # 这里简单假设从右下角拖拽，实际中左上角固定
            new_rect = [x0, y0, x1 + dx_base, y1 + dy_base]
            self.paint_save[self.selected_rect_index] = new_rect
            self.Show_Exists()

    def paintEvent(self, event):
        super(MainFunc, self).paintEvent(event)
        painter = QPainter(self.ui.label_4)
        painter.setPen(QPen(Qt.red, 4, Qt.SolidLine))
        # 绘制正在新建的矩形预览
        if self.flag and self.x0 != 0 and self.y0 != 0 and self.x1 != 0 and self.y1 != 0:
            painter.drawRect(QRect(self.x0, self.y0, abs(self.x1 - self.x0), abs(self.y1 - self.y0)))

    def saveAndUpdate(self):
        try:

            # 获取当前label上的QPixmap对象
            if self.ui.label_3.pixmap():
                # 当前实现中，新建矩形预览只在 label_4 上绘制，
                # 最终的持久化由 Show_Exists 统一在图像上重绘，这里不再额外处理。
                pass
        except Exception as e:
            print(f"Error saving and updating image: {e}")

# ##################################################################################################
    # 获取视频
    def get_video(self):
        self.ui.listWidget.clear()  # 清空listWidget
        self.image_files = None
        self.img_path = None
        self.num = 0
        video_save_path = QtWidgets.QFileDialog.getExistingDirectory(self, "选择图片保存文件夹")
        if video_save_path:
            self.video_save_path = video_save_path
        
        video_path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, 
            "选择视频", 
            "", 
            "Video Files (*.mp4 *.mpg)"
        )
        
        if video_path and video_save_path:
            self.video_path = video_path  # 保存视频路径以供重播使用
            self.cap = cv2.VideoCapture(video_path)
            # 获取视频总帧数
            self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
            # 设置滑块范围
            self.ui.horizontalSlider.setRange(0, self.total_frames)
            self.timer_camera.start(33)
            self.timer_camera.timeout.connect(self.OpenFrame)
            # 初始禁用重播按钮
            self.ui.pushButton_4.setEnabled(False)
        
            self.Change_Enable(method="ShowVideo", state=True)
            self.Change_Enable(method="MakeTag", state=False)
            # 禁用开始检测打标按钮
            self.ui.pushButton_start_marking.setEnabled(False)
        else:
            upWindowsh("请先选择视频和保存路径")


    def OpenFrame(self):
        if not self.sld_video_pressed:  # 只在未拖动时更新
            ret, image = self.cap.read()
            if ret:
                # 更新当前帧位置
                self.current_frame = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))
                # 更新滑块位置
                self.ui.horizontalSlider.setValue(self.current_frame)
                
                # 调整视频帧大小
                height, width = image.shape[:2]
                ratio = 1300 / width
                new_width = 1300
                new_height = int(height * ratio)
                
                if new_height > 850:
                    ratio = 850 / new_height
                    new_height = 850
                    new_width = int(new_width * ratio)
                
                # 调整图像大小
                image = cv2.resize(image, (new_width, new_height))
                
                if len(image.shape) == 3:
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    vedio_img = QImage(image.data, new_width, new_height, image.strides[0], QImage.Format_RGB888)
                elif len(image.shape) == 1:
                    vedio_img = QImage(image.data, new_width, new_height, QImage.Format_Indexed8)
                else:
                    vedio_img = QImage(image.data, new_width, new_height, image.strides[0], QImage.Format_RGB888)
                self.vedio_img = vedio_img
                
                # 调整label大小以适应新的图像尺寸
                self.ui.label_3.setFixedSize(new_width, new_height)
                self.ui.label_3.setPixmap(QPixmap(self.vedio_img))
                self.ui.label_3.setScaledContents(True)
            else:
                self.cap.release()
                self.timer_camera.stop()
                # 视频结束时启用重播按钮
                self.ui.pushButton_4.setEnabled(True)


    def Btn_Start(self):
        try:
            # 尝试断开之前的连接
            self.timer_camera.timeout.disconnect(self.OpenFrame)
        except TypeError:
            # 如果没有连接，直接忽略错误
            pass
        # 重新连接并启动定时器
        self.timer_camera.timeout.connect(self.OpenFrame)
        self.timer_camera.start(33)

    def Btn_Stop(self):
        self.timer_camera.stop()
        try:
            # 尝试断开定时器连接
            self.timer_camera.timeout.disconnect(self.OpenFrame)
        except TypeError:
            pass

    def Btn_Save(self):
        self.num += 1
        save_path = f'{self.video_save_path}/image{str(self.num)}.jpg'
        self.vedio_img.save(save_path)
        # 将保存信息添加到listWidget
        save_info = f'image{str(self.num)}.jpg保存成功！'
        self.ui.listWidget.addItem(save_info)
        print(f'{save_path}保存成功！')

    
    def moveSlider(self, position):
        """处理滑块移动"""
        if self.cap and self.total_frames > 0:
            # 设置视频帧位置
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, position)
            # 读取并显示新位置的帧
            ret, image = self.cap.read()
            if ret:
                # 调整视频帧大小
                height, width = image.shape[:2]
                ratio = 1300 / width
                new_width = 1300
                new_height = int(height * ratio)
                
                if new_height > 850:
                    ratio = 850 / new_height
                    new_height = 850
                    new_width = int(new_width * ratio)
                
                # 调整图像大小
                image = cv2.resize(image, (new_width, new_height))
                
                if len(image.shape) == 3:
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    vedio_img = QImage(image.data, new_width, new_height, image.strides[0], QImage.Format_RGB888)
                elif len(image.shape) == 1:
                    vedio_img = QImage(image.data, new_width, new_height, QImage.Format_Indexed8)
                else:
                    vedio_img = QImage(image.data, new_width, new_height, image.strides[0], QImage.Format_RGB888)
                self.vedio_img = vedio_img
                
                # 调整label大小以适应新的图像尺寸
                self.ui.label_3.setFixedSize(new_width, new_height)
                self.ui.label_3.setPixmap(QPixmap(self.vedio_img))
                self.ui.label_3.setScaledContents(True)

    def pressSlider(self):
        self.sld_video_pressed = True
        self.timer_camera.stop()  # 暂停视频播放

    def releaseSlider(self):
        self.sld_video_pressed = False
        try:
            # 尝试断开之前的连接
            self.timer_camera.timeout.disconnect(self.OpenFrame)
        except TypeError:
            pass
        # 重新连接并启动定时器
        self.timer_camera.timeout.connect(self.OpenFrame)
        self.timer_camera.start(33)

    def clean_up(self):
        file_path = 'GUI/history.txt'
        if os.path.exists(file_path):
            os.remove(file_path)

    # 删除当前图像中选中的标签（同时更新 XML 和显示）
    def delete_selected_label(self):
        if not self.save_path or not self.image_name:
            upWindowsh("当前没有可删除的标签文件")
            return
        xml_file = os.path.join(self.save_path, f"{self.image_name}.xml")
        if not os.path.exists(xml_file):
            upWindowsh("当前图像还没有标签文件")
            return
        row = self.ui.listWidget.currentRow()
        if row < 0:
            upWindowsh("请先在右侧列表中选择要删除的标签")
            return
        try:
            self.push_undo()
            tree = ET.parse(xml_file)
            root = tree.getroot()
            objects = root.findall("object")
            if row >= len(objects):
                upWindowsh("内部索引错误，无法删除该标签")
                return
            root.remove(objects[row])
            # 如果删除后没有任何 object，则直接删除 XML 文件
            if len(root.findall("object")) == 0:
                os.remove(xml_file)
            else:
                indent(root)
                tree.write(xml_file)

            # 重新加载并刷新显示
            self.labels = []
            self.clicked_save = []
            self.paint_save = []
            self.ui.listWidget.clear()
            self.Exists_Labels_And_Boxs()
            self.Show_Exists()
        except Exception as e:
            print(f"删除标签失败: {e}")
            upWindowsh("删除标签失败，请查看控制台输出")

    # 清空当前图像的所有标签（等价于删除对应 XML）
    def clear_all_labels(self):
        if not self.save_path or not self.image_name:
            upWindowsh("当前没有可清空的标签文件")
            return
        self.push_undo()
        xml_file = os.path.join(self.save_path, f"{self.image_name}.xml")
        if os.path.exists(xml_file):
            try:
                os.remove(xml_file)
            except Exception as e:
                print(f"清空标签失败: {e}")
                upWindowsh("清空标签失败，请查看控制台输出")
                return

        self.clicked_event = False
        self.paint_event = False
        self.save = True
        self.ui.listWidget.clear()
        self.list_labels = []
        self.clicked_save = []
        self.paint_save = []
        self.show_qt(self.img_path)
        self.ui.label_4.mousePressEvent = self.mouse_press_event
        self.ui.label_4.setCursor(Qt.ArrowCursor)

    def open_tag_management(self):
        """打开标签管理对话框，管理 history.txt 中的标签"""
        dlg = TagManagementDialog(self)
        dlg.exec_()

    def push_undo(self):
        """在修改当前图像标注前，将当前状态压入撤回栈（仅当有图像且已设保存路径时）。"""
        if not getattr(self, "image_name", None) or not getattr(self, "save_path", None):
            return
        self.undo_stack.append({
            "image_name": self.image_name,
            "labels": copy.deepcopy(self.labels),
            "clicked_save": copy.deepcopy(self.clicked_save),
            "paint_save": copy.deepcopy(self.paint_save),
        })

    def undo(self):
        """Ctrl+Z：撤回到上一步（仅对当前图像有效）。"""
        if not self.undo_stack or not getattr(self, "image_name", None) or not getattr(self, "save_path", None):
            return
        state = self.undo_stack.pop()
        if state["image_name"] != self.image_name:
            self.undo_stack.append(state)
            return
        self.labels = state["labels"]
        self.clicked_save = state["clicked_save"]
        self.paint_save = state["paint_save"]
        self.list_labels = [item.get("name", "") for item in self.labels]
        self.ui.listWidget.clear()
        for name in self.list_labels:
            self.ui.listWidget.addItem(name)
        # 写回 XML
        xml_file = os.path.join(self.save_path, f"{self.image_name}.xml")
        if not self.labels:
            if os.path.exists(xml_file):
                os.remove(xml_file)
        else:
            xml(self.image_path, xml_file, [self.img_width, self.img_height, 3], self.labels)
        self.Show_Exists()

    def Btn_Replay(self):
        """重新播放视频"""
        if hasattr(self, 'video_path'):
            # 重新打开视频文件
            self.cap = cv2.VideoCapture(self.video_path)
            # 重置滑块位置
            self.ui.horizontalSlider.setValue(0)
            # 开始播放
            self.timer_camera.start(33)                 
            # 禁用重播按钮
            self.ui.pushButton_4.setEnabled(False)

    def Btn_Auto(self):
        if self.video_path and self.video_save_path:
            output_dir,saved_count = self.AVT.extract_frames_from_video(self.video_path, self.video_save_path, fps=2)
            content = f"已从视频中提取 {saved_count} 帧\n保存至 {output_dir}"
            print(content)
            self.ui.listWidget.addItem(content)
        else:
            upWindowsh("请先选择视频和保存路径")

    def video_marking(self):
        self.directory = None
        video_path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "选择视频", "", "Video Files (*.mp4 *.mpg)")
        self.video_path = video_path

        output_dir = QtWidgets.QFileDialog.getExistingDirectory(self, "选择图片保存文件夹")
        self.output_dir = output_dir
        self.ui.listWidget.clear()
        if self.video_path and self.output_dir:
            self.Change_Enable(method="MakeTag",state=False)
            self.Change_Enable(method="ShowVideo",state=False)
            if self.cap:
                self.cap.release()
                self.timer_camera.stop()

            # 读取视频第一帧
            cap = cv2.VideoCapture(video_path)
            ret, frame = cap.read()
            if ret:
                cv2.imwrite(f"{self.output_dir}/0.jpg", frame)
            cap.release()
            if self.output_dir:
                self.image_files = list_images_in_directory(self.output_dir)
                if self.image_files:
                    self.image_path = self.image_files[0]
                    # print(self.image_path)
                    self.img_path = self.image_path
                    self.image_name = os.path.basename(self.image_path).split('.')[0]
                    # 更新 UI 显示当前图像名称并在后台打印
                    base_name = os.path.basename(self.image_path)
                    self.ui.label_image_name.setText(f"当前图像：{base_name}")
                    print(f"当前标注图像：{base_name}")

                    self.img_path, self.img_width, self.img_height = Change_image_Size(self.img_path)
                    print(self.img_path, self.img_width, self.img_height)
                    self.image = cv2.imread(self.img_path)
                    self.AT.Set_Image(self.image)
                    # 保存当前 CV 图像并显示
                    self.current_cv_image = self.image.copy()
                    self.show_qt(self.img_path)
                
            # 鼠标点击触发
            self.ui.label_4.mousePressEvent = self.mouse_press_event
        else:
            upWindowsh("请先选择视频和保存路径")


    def on_video_processing_complete(self):
        self.worker_thread.deleteLater()
        self.xml_messages = self.worker_thread.xml_messages
        # print(self.xml_messages)
        
        # 遍历输出目录中的图片
        for img_file in os.listdir(self.output_dir):
            if img_file.endswith(('.jpg', '.jpeg', '.png')):  # 检查图片文件扩展名
                # 获取不带扩展名的文件名
                img_name = os.path.splitext(img_file)[0]
                img_file  = os.path.join(self.output_dir,img_file)
                
                # 在xml_messages中查找对应的消息
                for msg in self.xml_messages:
                    self.labels = []
                    if len(msg) > 1:  # 确保msg有足够的元素
                        xml_path = msg[1]  # 获取索引值为1的路径
                        xml_filename = os.path.splitext(os.path.basename(xml_path))[0]
                        
                        # 如果文件名匹配，则复制XML文件到save_path
                        if xml_filename == img_name and self.save_path:
                            result = msg[0]
                            file_path = msg[1]
                            size = msg[2]
                            self.labels.append(result)
                            xml(img_file, file_path, size, self.labels)
        self.ui.listWidget.addItem("检测打标完成！")
        print("检测打标完成！")
                            

    def Btn_Start_Marking(self):
        # 禁用开始检测打标按钮
        self.ui.pushButton_start_marking.setEnabled(False)
        if self.video_path and self.output_dir:
            # 创建并启动工作线程
            self.worker_thread = VideoProcessingThread(self.AVT, self.video_path, self.output_dir,self.clicked_x, self.clicked_y, self.method,self.text,self.save_path)
            self.worker_thread.finished.connect(self.on_video_processing_complete)
            self.worker_thread.start()
            

        else:
            upWindowsh("请先选择视频和保存路径")

        

