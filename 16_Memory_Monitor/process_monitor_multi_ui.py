import sys
import os
import time
from datetime import datetime
import numpy as np
import pandas as pd
import psutil
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
    QLabel, QLineEdit, QPushButton, QFileDialog, QMessageBox, QTableWidget,
    QTableWidgetItem, QHeaderView, QTabWidget, QSplitter, QStatusBar, QFrame,
    QComboBox, QListWidget, QListWidgetItem, QDialog, QDialogButtonBox, QAbstractItemView
)
from PyQt5.QtCore import Qt, QTimer, QThread, pyqtSignal, QSize, QSettings
from PyQt5.QtGui import QFont, QPalette, QColor, QIcon, QBrush, QLinearGradient, QPainter, QPen, QPixmap
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import matplotlib.style as mplstyle
from matplotlib.ticker import FuncFormatter
from collections import deque, defaultdict
import platform

mplstyle.use('fast')

COLORS = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
          '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']


class ProcessSearchDialog(QDialog):
    """进程搜索对话框"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("查找进程")
        self.setModal(True)
        self.resize(800, 400)

        layout = QVBoxLayout(self)

        # 搜索区域
        search_layout = QHBoxLayout()
        search_layout.addWidget(QLabel("搜索:"))
        self.search_input = QLineEdit()
        self.search_input.textChanged.connect(self.filter_processes)
        search_layout.addWidget(self.search_input)
        layout.addLayout(search_layout)

        # 进程表格
        self.process_table = QTableWidget()
        self.process_table.setColumnCount(3)
        self.process_table.setHorizontalHeaderLabels(["PID", "进程名", "内存使用(MB)"])
        self.process_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.process_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.process_table.setSelectionMode(QAbstractItemView.SingleSelection)
        self.process_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        layout.addWidget(self.process_table)

        # 按钮
        button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)
        layout.addWidget(button_box)

        # 加载进程列表
        self.load_processes()

    def load_processes(self):
        """加载当前运行的进程列表"""
        self.process_table.setRowCount(0)

        try:
            for proc in psutil.process_iter(['pid', 'name', 'memory_info']):
                try:
                    pid = proc.info['pid']
                    name = proc.info['name']
                    mem_info = proc.info['memory_info']
                    rss_mb = mem_info.rss / (1024 * 1024) if mem_info else 0

                    row_position = self.process_table.rowCount()
                    self.process_table.insertRow(row_position)

                    # PID
                    pid_item = QTableWidgetItem(str(pid))
                    pid_item.setData(Qt.UserRole, pid)
                    self.process_table.setItem(row_position, 0, pid_item)

                    # 进程名
                    name_item = QTableWidgetItem(name)
                    self.process_table.setItem(row_position, 1, name_item)

                    # 内存使用
                    mem_item = QTableWidgetItem(f"{rss_mb:.1f}")
                    self.process_table.setItem(row_position, 2, mem_item)

                except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                    continue
        except Exception as e:
            QMessageBox.warning(self, "错误", f"加载进程列表失败: {str(e)}")

    def filter_processes(self, text):
        """过滤进程列表"""
        for row in range(self.process_table.rowCount()):
            pid_item = self.process_table.item(row, 0)
            name_item = self.process_table.item(row, 1)

            if pid_item and name_item:
                pid = pid_item.text()
                name = name_item.text().lower()
                visible = text.lower() in name or text in pid
                self.process_table.setRowHidden(row, not visible)

    def get_selected_process(self):
        """获取选中的进程"""
        selected_items = self.process_table.selectedItems()
        if not selected_items:
            return None

        row = selected_items[0].row()
        pid_item = self.process_table.item(row, 0)
        name_item = self.process_table.item(row, 1)

        if pid_item and name_item:
            pid = int(pid_item.text())
            name = name_item.text()
            return {'pid': pid, 'name': name}

        return None


class MultiProcessMonitorThread(QThread):
    """多进程监控线程"""
    data_updated = pyqtSignal(dict)
    process_ended = pyqtSignal(int)
    error_occurred = pyqtSignal(str)

    def __init__(self, monitored_processes, interval=1.0):
        super().__init__()
        self.monitored_processes = monitored_processes  # {pid: {'name': name, 'type': 'name'|'pid'}, ...}
        self.interval = interval
        self.running = False
        self.process_handles = {}  # {pid: psutil.Process object, ...}

    def setup_processes(self):
        """设置要监控的进程"""
        for pid, proc_info in self.monitored_processes.items():
            try:
                if proc_info['type'] == 'pid':
                    # 通过PID查找
                    self.process_handles[pid] = psutil.Process(pid)
                else:
                    # 通过名称查找，可能有多个同名进程
                    found = False
                    for proc in psutil.process_iter(['pid', 'name']):
                        try:
                            if proc_info['name'].lower() in proc.info['name'].lower():
                                self.process_handles[proc.info['pid']] = proc
                                found = True
                                break
                        except (psutil.NoSuchProcess, psutil.AccessDenied):
                            continue

                    if not found:
                        self.error_occurred.emit(f"未找到名称包含 '{proc_info['name']}' 的进程")
                        return False
            except psutil.NoSuchProcess:
                self.error_occurred.emit(f"未找到PID为 {pid} 的进程")
                return False
            except Exception as e:
                self.error_occurred.emit(f"查找进程时出错: {str(e)}")
                return False

        return True

    def run(self):
        """线程主函数"""
        self.running = True

        if not self.setup_processes():
            self.running = False
            return

        while self.running:
            current_time = datetime.now()
            update_data = {}

            for pid, proc in list(self.process_handles.items()):
                try:
                    if not proc.is_running() or proc.status() == psutil.STATUS_ZOMBIE:
                        self.process_ended.emit(pid)
                        del self.process_handles[pid]
                        continue

                    # 获取内存信息
                    mem_info = proc.memory_info()
                    mem_percent = proc.memory_percent()
                    cpu_percent = proc.cpu_percent(interval=0.1)

                    # 保存数据
                    update_data[pid] = {
                        'timestamp': current_time,
                        'pid': pid,
                        'name': proc.name(),
                        'memory_percent': mem_percent,
                        'memory_rss': mem_info.rss / (1024 * 1024),  # MB
                        'memory_vms': mem_info.vms / (1024 * 1024),  # MB
                        'cpu_percent': cpu_percent
                    }

                except psutil.NoSuchProcess:
                    self.process_ended.emit(pid)
                    if pid in self.process_handles:
                        del self.process_handles[pid]
                except Exception as e:
                    self.error_occurred.emit(f"监控PID={pid}时出错: {str(e)}")

            if update_data:
                self.data_updated.emit(update_data)

            time.sleep(self.interval)

    def stop(self):
        """停止监控"""
        self.running = False
        self.wait()


class MemoryChart(FigureCanvas):
    """内存使用图表，支持多条线"""

    def __init__(self, parent=None, width=5, height=4, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = self.fig.add_subplot(111)

        super().__init__(self.fig)
        self.setParent(parent)

        # 数据存储
        self.process_data = {}  # {pid: {'timestamps': [], 'memory': [], 'name': '', 'color': ''}, ...}
        self.max_points = 300

        # 配置图表
        self.configure_plot()

    def configure_plot(self):
        """配置图表样式"""
        self.axes.set_title('Memory usage situation', fontsize=12)
        self.axes.set_xlabel('Time', fontsize=10)
        self.axes.set_ylabel('Memory (MB)', fontsize=10)
        self.axes.grid(True, linestyle='--', alpha=0.7)
        self.axes.tick_params(axis='both', which='major', labelsize=8)

        # 设置x轴日期格式
        self.axes.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
        self.axes.xaxis.set_major_locator(mdates.AutoDateLocator())

        # 紧凑布局
        self.fig.tight_layout()

    def add_process(self, pid, name, color=None):
        """添加要监控的进程"""
        if pid in self.process_data:
            return

        # 选择颜色
        if color is None:
            color_index = len(self.process_data) % len(COLORS)
            color = COLORS[color_index]

        self.process_data[pid] = {
            'timestamps': deque(maxlen=self.max_points),
            'memory': deque(maxlen=self.max_points),
            'name': name,
            'color': color
        }

    def remove_process(self, pid):
        """移除进程"""
        if pid in self.process_data:
            del self.process_data[pid]
            self.redraw_chart()

    def update_data(self, pid, timestamp, memory_value):
        """更新指定进程的数据"""
        if pid not in self.process_data:
            return

        self.process_data[pid]['timestamps'].append(timestamp)
        self.process_data[pid]['memory'].append(memory_value)
        self.redraw_chart()

    def clear_all_data(self):
        """清除所有数据"""
        for pid in self.process_data:
            self.process_data[pid]['timestamps'].clear()
            self.process_data[pid]['memory'].clear()
        self.redraw_chart()

    def redraw_chart(self):
        """重新绘制整个图表"""
        self.axes.clear()
        self.configure_plot()

        # 找出全局最大内存值，用于设置Y轴
        all_memory_values = []
        for pid, data in self.process_data.items():
            if data['memory']:
                all_memory_values.extend(data['memory'])

        if not all_memory_values:
            self.draw()
            return

        # 设置Y轴范围
        y_max = max(all_memory_values) * 1.1
        if y_max < 10:
            y_max = 10
        self.axes.set_ylim(0, y_max)

        # 为每个进程绘制线条
        has_data = False
        for pid, data in self.process_data.items():
            if data['timestamps'] and data['memory']:
                has_data = True
                timestamps = list(data['timestamps'])
                memory_values = list(data['memory'])

                # 绘制线条
                line = self.axes.plot(timestamps, memory_values,
                                      color=data['color'], linewidth=2,
                                      label=f"{data['name']} (PID: {pid})")[0]

                # 为最新点添加标记
                if len(timestamps) > 0:
                    latest_time = timestamps[-1]
                    latest_value = memory_values[-1]
                    self.axes.scatter(latest_time, latest_value,
                                      color=data['color'], s=40, zorder=5)
                    # 添加标签
                    self.axes.annotate(f"{latest_value:.1f} MB",
                                       xy=(latest_time, latest_value),
                                       xytext=(5, 5), textcoords='offset points',
                                       fontsize=8, bbox=dict(boxstyle="round,pad=0.3",
                                                             fc="white", ec=data['color'], alpha=0.8))

        if has_data:
            # 添加图例
            self.axes.legend(loc='upper left', fontsize=8, framealpha=0.8)

            # 自动旋转x轴标签
            self.fig.autofmt_xdate()

        # 刷新图表
        self.draw()

    def get_all_data(self):
        """获取所有进程的完整数据，用于保存"""
        all_data = []

        for pid, data in self.process_data.items():
            for i in range(len(data['timestamps'])):
                all_data.append({
                    'timestamp': data['timestamps'][i],
                    'pid': pid,
                    'name': data['name'],
                    'memory_rss': data['memory'][i]
                })

        return all_data


class MonitoredProcessItem(QWidget):
    """监控进程列表中的单个项目"""
    remove_requested = pyqtSignal(int)

    def __init__(self, pid, name, parent=None):
        super().__init__(parent)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(5, 3, 5, 3)

        # 进程信息
        info_layout = QVBoxLayout()
        info_layout.setSpacing(0)

        name_label = QLabel(f"<b>{name}</b>")
        name_label.setStyleSheet("font-weight: bold;")
        pid_label = QLabel(f"PID: {pid}")
        pid_label.setStyleSheet("color: #666; font-size: 9pt;")

        info_layout.addWidget(name_label)
        info_layout.addWidget(pid_label)

        layout.addLayout(info_layout, 1)

        # 移除按钮
        remove_btn = QPushButton("×")
        remove_btn.setFixedSize(24, 24)
        remove_btn.setStyleSheet("""
            QPushButton {
                background-color: #ff6b6b;
                color: white;
                border-radius: 12px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #ee5a5a;
            }
        """)
        remove_btn.clicked.connect(lambda: self.remove_requested.emit(pid))
        layout.addWidget(remove_btn)

        self.pid = pid
        self.name = name


class MultiProcessMonitorUI(QMainWindow):
    """多进程监控主窗口"""

    def __init__(self):
        super().__init__()

        # 窗口设置
        self.setWindowTitle("多进程内存监控工具")
        self.setGeometry(100, 100, 1200, 800)

        # 数据存储
        self.monitored_processes = {}  # {pid: {'name': name, 'type': 'name'|'pid'}, ...}
        self.monitoring = False
        self.all_monitor_data = defaultdict(list)  # {pid: [data_points], ...}

        # 加载设置
        self.settings = QSettings("ProcessMonitor", "MultiProcessMonitor")
        self.load_settings()

        # 创建UI
        self.init_ui()

        # 创建监控线程
        self.monitor_thread = None

    def init_ui(self):
        """初始化用户界面"""
        # 主窗口部件
        main_widget = QWidget()
        self.setCentralWidget(main_widget)

        # 主布局
        main_layout = QVBoxLayout(main_widget)

        # 顶部控制区域
        control_frame = QFrame()
        control_frame.setFrameShape(QFrame.StyledPanel)
        control_layout = QVBoxLayout(control_frame)

        # 标题
        title_label = QLabel("多进程内存监控工具")
        title_label.setFont(QFont("Arial", 18, QFont.Bold))
        title_label.setAlignment(Qt.AlignCenter)
        title_label.setStyleSheet("color: #2c3e50; margin: 10px 0;")
        control_layout.addWidget(title_label)

        # 添加进程区域
        add_process_layout = QHBoxLayout()

        self.search_type_combo = QComboBox()
        self.search_type_combo.addItems(["按进程名", "按PID", "从列表选择"])
        add_process_layout.addWidget(QLabel("添加方式:"))
        add_process_layout.addWidget(self.search_type_combo, 1)

        self.process_input = QLineEdit()
        self.process_input.setPlaceholderText("输入进程名或PID")
        add_process_layout.addWidget(self.process_input, 2)

        add_btn = QPushButton("添加进程")
        add_btn.setStyleSheet("background-color: #3498db; color: white; font-weight: bold;")
        add_btn.clicked.connect(self.add_process)
        add_process_layout.addWidget(add_btn)

        control_layout.addLayout(add_process_layout)

        # 监控进程列表
        list_layout = QVBoxLayout()
        list_layout.addWidget(QLabel("监控列表:"))

        self.process_list = QListWidget()
        self.process_list.setSelectionMode(QAbstractItemView.NoSelection)
        list_layout.addWidget(self.process_list, 1)

        control_layout.addLayout(list_layout)

        # 控制按钮
        button_layout = QHBoxLayout()

        self.start_btn = QPushButton("开始监控")
        self.start_btn.setStyleSheet("background-color: #2ecc71; color: white; font-weight: bold; font-size: 11pt;")
        self.start_btn.setMinimumHeight(40)
        self.start_btn.clicked.connect(self.start_monitoring)

        self.stop_btn = QPushButton("停止监控")
        self.stop_btn.setStyleSheet("background-color: #e74c3c; color: white; font-weight: bold; font-size: 11pt;")
        self.stop_btn.setMinimumHeight(40)
        self.stop_btn.setEnabled(False)
        self.stop_btn.clicked.connect(self.stop_monitoring)

        self.save_btn = QPushButton("保存数据")
        self.save_btn.setStyleSheet("background-color: #3498db; color: white; font-weight: bold;")
        self.save_btn.setEnabled(False)
        self.save_btn.clicked.connect(self.save_data)

        self.clear_btn = QPushButton("清除数据")
        self.clear_btn.setStyleSheet("background-color: #95a5a6; color: white; font-weight: bold;")
        self.clear_btn.setEnabled(False)
        self.clear_btn.clicked.connect(self.clear_data)

        button_layout.addWidget(self.start_btn, 2)
        button_layout.addWidget(self.stop_btn, 2)
        button_layout.addWidget(self.save_btn, 1)
        button_layout.addWidget(self.clear_btn, 1)

        control_layout.addLayout(button_layout)

        # 间隔和点数控制
        config_layout = QHBoxLayout()

        config_layout.addWidget(QLabel("监控间隔(秒):"))
        self.interval_input = QLineEdit("1.0")
        self.interval_input.setFixedWidth(60)
        config_layout.addWidget(self.interval_input)

        config_layout.addWidget(QLabel("保留数据点数:"))
        self.max_points_input = QLineEdit("300")
        self.max_points_input.setFixedWidth(60)
        config_layout.addWidget(self.max_points_input)

        config_layout.addStretch()
        control_layout.addLayout(config_layout)

        main_layout.addWidget(control_frame)

        # 图表区域
        chart_frame = QFrame()
        chart_frame.setFrameShape(QFrame.StyledPanel)
        chart_layout = QVBoxLayout(chart_frame)

        # 创建图表
        self.memory_chart = MemoryChart(self, width=5, height=4, dpi=100)
        chart_layout.addWidget(self.memory_chart)

        main_layout.addWidget(chart_frame, 1)

        # 状态栏
        self.statusBar = QStatusBar()
        self.statusBar.setStyleSheet("QStatusBar{padding: 5px; background: #f8f9fa;}")
        self.setStatusBar(self.statusBar)
        self.statusBar.showMessage("就绪 | 选择要监控的进程")

        # 设置窗口图标
        self.set_window_icon()

    def set_window_icon(self):
        """设置窗口图标（使用内置图标）"""
        try:
            from PyQt5.QtGui import QIcon
            # 创建一个简单的图标
            icon = QIcon()
            pixmap = QPixmap(32, 32)
            pixmap.fill(Qt.transparent)

            painter = QPainter(pixmap)
            painter.setRenderHint(QPainter.Antialiasing)

            # 绘制一个简单的图表图标
            painter.setPen(QPen(QColor("#3498db"), 2))
            painter.drawLine(5, 25, 10, 15)
            painter.drawLine(10, 15, 15, 20)
            painter.drawLine(15, 20, 20, 10)
            painter.drawLine(20, 10, 25, 18)

            # 绘制坐标轴
            painter.setPen(QPen(QColor("#2c3e50"), 1))
            painter.drawLine(5, 27, 27, 27)  # X轴
            painter.drawLine(5, 27, 5, 5)  # Y轴

            painter.end()

            icon.addPixmap(pixmap)
            self.setWindowIcon(icon)
        except:
            pass

    def add_process(self):
        """添加要监控的进程"""
        search_type = self.search_type_combo.currentIndex()
        input_text = self.process_input.text().strip()

        if search_type == 0:  # 按进程名
            if not input_text:
                QMessageBox.warning(self, "输入错误", "请输入进程名")
                return

            # 检查是否有同名进程已经在监控
            for pid, proc_info in self.monitored_processes.items():
                if proc_info['type'] == 'name' and proc_info['name'] == input_text:
                    QMessageBox.information(self, "提示", f"进程 '{input_text}' 已在监控列表中")
                    return

            # 添加到监控列表
            dummy_pid = hash(input_text) % 100000  # 生成一个伪PID用于标识
            while dummy_pid in self.monitored_processes:
                dummy_pid += 1

            self.monitored_processes[dummy_pid] = {
                'name': input_text,
                'type': 'name'
            }

            self.add_process_to_list(dummy_pid, input_text)
            self.process_input.clear()

        elif search_type == 1:  # 按PID
            if not input_text or not input_text.isdigit():
                QMessageBox.warning(self, "输入错误", "请输入有效的PID数字")
                return

            pid = int(input_text)
            if pid in self.monitored_processes:
                QMessageBox.information(self, "提示", f"PID {pid} 已在监控列表中")
                return

            try:
                proc = psutil.Process(pid)
                name = proc.name()

                self.monitored_processes[pid] = {
                    'name': name,
                    'type': 'pid'
                }

                self.add_process_to_list(pid, name)
                self.process_input.clear()
            except psutil.NoSuchProcess:
                QMessageBox.warning(self, "错误", f"未找到PID为 {pid} 的进程")
            except Exception as e:
                QMessageBox.warning(self, "错误", f"获取进程信息失败: {str(e)}")

        else:  # 从列表选择
            dialog = ProcessSearchDialog(self)
            if dialog.exec_() == QDialog.Accepted:
                process_info = dialog.get_selected_process()
                if process_info:
                    pid = process_info['pid']
                    name = process_info['name']

                    if pid in self.monitored_processes:
                        QMessageBox.information(self, "提示", f"PID {pid} 已在监控列表中")
                        return

                    self.monitored_processes[pid] = {
                        'name': name,
                        'type': 'pid'
                    }

                    self.add_process_to_list(pid, name)

    def add_process_to_list(self, pid, name):
        """将进程添加到UI列表"""
        item_widget = MonitoredProcessItem(pid, name)
        item_widget.remove_requested.connect(self.remove_process)

        item = QListWidgetItem()
        item.setSizeHint(item_widget.sizeHint())

        self.process_list.addItem(item)
        self.process_list.setItemWidget(item, item_widget)

        # 为图表添加进程
        if self.monitoring:
            self.memory_chart.add_process(pid, name)

        self.statusBar.showMessage(f"已添加进程: {name} (PID: {pid})")

    def remove_process(self, pid):
        """从监控列表中移除进程"""
        reply = QMessageBox.question(self, '确认移除',
                                     '确定要移除此进程吗?',
                                     QMessageBox.Yes | QMessageBox.No,
                                     QMessageBox.No)

        if reply == QMessageBox.Yes:
            # 从监控列表中移除
            if pid in self.monitored_processes:
                del self.monitored_processes[pid]

            # 从UI列表中移除
            for i in range(self.process_list.count()):
                item = self.process_list.item(i)
                widget = self.process_list.itemWidget(item)
                if hasattr(widget, 'pid') and widget.pid == pid:
                    self.process_list.takeItem(i)
                    break

            # 从图表中移除
            if self.monitoring:
                self.memory_chart.remove_process(pid)
                if pid in self.all_monitor_data:
                    del self.all_monitor_data[pid]

            self.statusBar.showMessage(f"已移除进程 PID: {pid}")

    def start_monitoring(self):
        """开始监控所有进程"""
        if not self.monitored_processes:
            QMessageBox.warning(self, "错误", "请先添加要监控的进程")
            return

        if self.monitoring:
            return

        # 获取监控参数
        try:
            interval = float(self.interval_input.text())
            if interval <= 0:
                raise ValueError("间隔必须大于0")

            max_points = int(self.max_points_input.text())
            if max_points <= 0:
                raise ValueError("数据点数必须大于0")
        except ValueError as e:
            QMessageBox.warning(self, "输入错误", f"请输入有效的参数: {str(e)}")
            return

        # 设置图表的最大点数
        self.memory_chart.max_points = max_points
        self.memory_chart.clear_all_data()

        # 重置监控数据
        self.all_monitor_data = defaultdict(list)

        # 为每个进程在图表中添加条目
        for pid, proc_info in self.monitored_processes.items():
            self.memory_chart.add_process(pid, proc_info['name'])

        # 创建并启动监控线程
        self.monitor_thread = MultiProcessMonitorThread(
            monitored_processes=self.monitored_processes.copy(),
            interval=interval
        )

        # 连接信号
        self.monitor_thread.data_updated.connect(self.update_data)
        self.monitor_thread.process_ended.connect(self.handle_process_ended)
        self.monitor_thread.error_occurred.connect(self.handle_error)

        # 更新UI状态
        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.save_btn.setEnabled(False)
        self.clear_btn.setEnabled(False)
        self.monitoring = True

        # 启动线程
        self.monitor_thread.start()
        self.statusBar.showMessage(f"开始监控 {len(self.monitored_processes)} 个进程...")

    def stop_monitoring(self):
        """停止监控"""
        if self.monitor_thread and self.monitoring:
            self.monitor_thread.stop()
            self.monitor_thread = None
            self.monitoring = False

        # 更新UI状态
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.save_btn.setEnabled(True)
        self.clear_btn.setEnabled(True)

        self.statusBar.showMessage(
            f"监控已停止 | 共收集 {sum(len(data) for data in self.all_monitor_data.values())} 条数据")

    def update_data(self, update_data):
        """更新图表和数据"""
        current_time = datetime.now()

        for pid, data in update_data.items():
            # 保存数据
            self.all_monitor_data[pid].append(data)

            # 更新图表
            if pid in self.memory_chart.process_data:
                self.memory_chart.update_data(pid, data['timestamp'], data['memory_rss'])

    def handle_process_ended(self, pid):
        """处理进程结束事件"""
        # 从监控列表中查找进程名
        name = "未知进程"
        if pid in self.monitored_processes:
            name = self.monitored_processes[pid]['name']

        # 显示提示
        self.statusBar.showMessage(f"进程 {name} (PID: {pid}) 已结束", 5000)

        # 如果所有进程都结束了，自动停止监控
        if self.monitor_thread and self.monitor_thread.process_handles:
            remaining = len(self.monitor_thread.process_handles)
            if remaining == 0:
                self.stop_monitoring()
                QMessageBox.information(self, "监控结束", "所有被监控的进程都已结束运行。")

    def handle_error(self, error_message):
        """处理错误"""
        QMessageBox.warning(self, "监控错误", error_message)
        self.statusBar.showMessage(f"错误: {error_message}")

    def save_data(self):
        """保存所有监控数据到CSV"""
        if not self.all_monitor_data:
            QMessageBox.warning(self, "无数据", "没有监控数据可保存")
            return

        # 打开文件对话框
        file_path, _ = QFileDialog.getSaveFileName(
            self, "保存监控数据", "", "CSV文件 (*.csv);;所有文件 (*)"
        )

        if not file_path:
            return

        # 确保文件扩展名
        if not file_path.endswith('.csv'):
            file_path += '.csv'

        try:
            # 准备数据
            all_rows = []
            for pid, data_points in self.all_monitor_data.items():
                for data in data_points:
                    row = {
                        'timestamp': data['timestamp'].strftime('%Y-%m-%d %H:%M:%S.%f'),
                        'pid': data['pid'],
                        'name': data['name'],
                        'memory_percent': f"{data['memory_percent']:.2f}",
                        'memory_rss_mb': f"{data['memory_rss']:.2f}",
                        'memory_vms_mb': f"{data['memory_vms']:.2f}",
                        'cpu_percent': f"{data['cpu_percent']:.2f}"
                    }
                    all_rows.append(row)

            # 保存到CSV
            df = pd.DataFrame(all_rows)
            df.to_csv(file_path, index=False, encoding='utf-8-sig')

            QMessageBox.information(self, "保存成功", f"监控数据已保存到:\n{file_path}")
            self.statusBar.showMessage(f"数据已保存至 {file_path}")
        except Exception as e:
            QMessageBox.critical(self, "保存失败", f"保存数据时出错:\n{str(e)}")
            self.statusBar.showMessage(f"保存失败: {str(e)}")

    def clear_data(self):
        """清除所有监控数据"""
        reply = QMessageBox.question(self, '确认清除',
                                     '确定要清除所有监控数据吗? 这将重置图表显示。',
                                     QMessageBox.Yes | QMessageBox.No,
                                     QMessageBox.No)

        if reply == QMessageBox.Yes:
            self.all_monitor_data = defaultdict(list)
            self.memory_chart.clear_all_data()
            self.save_btn.setEnabled(False)
            self.statusBar.showMessage("已清除所有监控数据")

    def load_settings(self):
        """加载保存的设置"""
        # 这里可以加载之前监控的进程等设置
        pass

    def save_settings(self):
        """保存当前设置"""
        # 这里可以保存监控列表等设置
        pass

    def closeEvent(self, event):
        """处理窗口关闭事件"""
        self.stop_monitoring()
        self.save_settings()
        event.accept()


if __name__ == "__main__":
    # 检查依赖
    try:
        import psutil
        import matplotlib
        import pandas
    except ImportError as e:
        print(f"缺少必要的依赖库: {str(e)}")
        print("请安装所需库: pip install psutil matplotlib pandas numpy pyqt5")
        sys.exit(1)

    # 创建应用
    app = QApplication(sys.argv)

    # 设置应用样式
    app.setStyle("Fusion")

    # 设置调色板
    palette = QPalette()
    palette.setColor(QPalette.Window, QColor(240, 240, 240))
    palette.setColor(QPalette.WindowText, QColor(40, 40, 40))
    palette.setColor(QPalette.Base, QColor(255, 255, 255))
    palette.setColor(QPalette.AlternateBase, QColor(245, 245, 245))
    palette.setColor(QPalette.ToolTipBase, QColor(255, 255, 220))
    palette.setColor(QPalette.ToolTipText, QColor(0, 0, 0))
    palette.setColor(QPalette.Text, QColor(40, 40, 40))
    palette.setColor(QPalette.Button, QColor(255, 255, 255))
    palette.setColor(QPalette.ButtonText, QColor(40, 40, 40))
    palette.setColor(QPalette.BrightText, QColor(255, 0, 0))
    palette.setColor(QPalette.Highlight, QColor(70, 130, 180))
    palette.setColor(QPalette.HighlightedText, QColor(255, 255, 255))
    app.setPalette(palette)

    # 设置字体
    font = QFont("Arial", 10)
    app.setFont(font)

    # 创建并显示主窗口
    window = MultiProcessMonitorUI()
    window.show()

    # 运行应用
    sys.exit(app.exec_())