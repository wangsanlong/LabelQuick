# -*- coding: utf-8 -*-
"""标签管理对话框：新增和删除 history.txt 中的标签"""
import os
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QListWidget, QLineEdit,
    QPushButton, QLabel, QMessageBox
)
from util.QtFunc import upWindowsh

HISTORY_PATH = 'GUI/history.txt'


class TagManagementDialog(QDialog):
    def __init__(self, parent=None, stats=None):
        super(TagManagementDialog, self).__init__(parent)
        self.setWindowTitle("标签管理")
        self.setMinimumSize(320, 360)
        self.history = []
        self.stats = stats or {}
        self._setup_ui()
        self._load_history()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("历史标签列表（来自 history.txt）："))
        self.listWidget = QListWidget()
        self.listWidget.setSelectionMode(QListWidget.ExtendedSelection)
        layout.addWidget(self.listWidget)

        add_layout = QHBoxLayout()
        self.lineEdit = QLineEdit()
        self.lineEdit.setPlaceholderText("输入新标签名称")
        self.lineEdit.returnPressed.connect(self._add_tag)
        add_layout.addWidget(self.lineEdit)
        btn_add = QPushButton("新增")
        btn_add.clicked.connect(self._add_tag)
        add_layout.addWidget(btn_add)
        layout.addLayout(add_layout)

        layout.addWidget(QLabel("标签统计（当前图像）："))
        self.label_stats = QLabel("")
        self.label_stats.setTextInteractionFlags(Qt.TextSelectableByMouse)
        layout.addWidget(self.label_stats)

        btn_layout = QHBoxLayout()
        btn_delete = QPushButton("删除选中")
        btn_delete.clicked.connect(self._delete_selected)
        btn_layout.addWidget(btn_delete)
        btn_layout.addStretch()
        btn_close = QPushButton("关闭")
        btn_close.clicked.connect(self.accept)
        btn_layout.addWidget(btn_close)
        layout.addLayout(btn_layout)
        self._refresh_stats()

    def _refresh_stats(self):
        if not self.stats:
            self.label_stats.setText("无")
            return
        lines = [f"{k}: {v}" for k, v in sorted(self.stats.items(), key=lambda kv: (-kv[1], kv[0]))]
        self.label_stats.setText("\n".join(lines))

    def _load_history(self):
        self.history = []
        try:
            if os.path.exists(HISTORY_PATH):
                with open(HISTORY_PATH, 'r', encoding='utf-8') as f:
                    self.history = [line.strip() for line in f.readlines() if line.strip()]
        except Exception as e:
            upWindowsh(f"读取标签失败：{e}")
        self._refresh_list()

    def _refresh_list(self):
        self.listWidget.clear()
        for item in self.history:
            self.listWidget.addItem(item)

    def _save_history(self):
        try:
            os.makedirs(os.path.dirname(HISTORY_PATH), exist_ok=True)
            with open(HISTORY_PATH, 'w', encoding='utf-8') as f:
                for item in self.history:
                    f.write(f"{item}\n")
        except Exception as e:
            upWindowsh(f"保存标签失败：{e}")

    def _add_tag(self):
        text = self.lineEdit.text().strip()
        if not text:
            upWindowsh("请输入标签名称")
            return
        if text in self.history:
            upWindowsh("该标签已存在")
            return
        self.history.append(text)
        self._refresh_list()
        self._save_history()
        self.lineEdit.clear()

    def _delete_selected(self):
        rows = [i.row() for i in self.listWidget.selectedIndexes()]
        if not rows:
            upWindowsh("请先选择要删除的标签")
            return
        for row in sorted(rows, reverse=True):
            if 0 <= row < len(self.history):
                del self.history[row]
        self._refresh_list()
        self._save_history()
