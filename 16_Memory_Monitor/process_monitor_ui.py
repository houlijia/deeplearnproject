import sys
import os
import csv
import time
from datetime import datetime
import numpy as np
import pandas as pd
import psutil
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QLineEdit, QPushButton, QFileDialog, QMessageBox,
    QSplitter, QStatusBar, QFrame, QComboBox
)
from PyQt5.QtCore import Qt, QTimer, QThread, pyqtSignal, QSize
from PyQt5.QtGui import QFont, QPalette, QColor, QIcon
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.dates as mdates
from matplotlib.ticker import FuncFormatter
import matplotlib.style as mplstyle

mplstyle.use('fast')


class ProcessMonitorThread(QThread):
    """后台线程，用于监控进程数据"""
    data_updated = pyqtSignal(dict)
    process_ended = pyqtSignal()
    error_occurred = pyqtSignal(str)

    def __init__(self, process_name=None, pid=None, interval=1.0):
        super().__init__()
        self.process_name = process_name
        self.pid = pid
        self.interval = interval
        self.running = False
        self.process = None

    def find_process(self):
        """查找要监控的进程"""
        try:
            if self.pid is not None:
                self.process = psutil.Process(int(self.pid))
            elif self.process_name:
                for proc in psutil.process_iter(['pid', 'name']):
                    try:
                        if self.process_name.lower() in proc.info['name'].lower():
                            self.process = proc
                            break
                    except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                        continue

            if not self.process:
                self.error_occurred.emit(f"未找到名称包含 '{self.process_name}' 的进程")
                return False

            return True
        except psutil.NoSuchProcess:
            self.error_occurred.emit(f"未找到PID为 {self.pid} 的进程")
            return False
        except Exception as e:
            self.error_occurred.emit(f"查找进程时出错: {str(e)}")
            return False

    def run(self):
        """线程主函数，收集进程数据"""
        self.running = True

        if not self.find_process():
            self.running = False
            return

        while self.running:
            try:
                # 检查进程是否仍在运行
                if not self.process.is_running() or self.process.status() == psutil.STATUS_ZOMBIE:
                    self.process_ended.emit()
                    break

                # 获取内存信息
                mem_info = self.process.memory_info()
                mem_percent = self.process.memory_percent()
                cpu_percent = self.process.cpu_percent(interval=0.1)

                # 准备数据
                data = {
                    'timestamp': datetime.now(),
                    'pid': self.process.pid,
                    'name': self.process.name(),
                    'memory_percent': mem_percent,
                    'memory_rss': mem_info.rss / (1024 * 1024),  # 转为MB
                    'memory_vms': mem_info.vms / (1024 * 1024),  # 转为MB
                    'cpu_percent': cpu_percent
                }

                # 发送数据更新信号
                self.data_updated.emit(data)

                # 等待下一个采集周期
                time.sleep(self.interval)

            except psutil.NoSuchProcess:
                self.process_ended.emit()
                break
            except Exception as e:
                self.error_occurred.emit(f"监控出错: {str(e)}")
                time.sleep(self.interval)

    def stop(self):
        """停止监控"""
        self.running = False
        self.wait()


class MemoryChart(FigureCanvas):
    """内存使用图表"""

    def __init__(self, parent=None, width=5, height=4, dpi=100):
        fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = fig.add_subplot(111)

        super().__init__(fig)
        self.setParent(parent)

        # 数据存储
        self.timestamps = []
        self.memory_values = []
        self.max_points = 200

        # 配置图表
        self.configure_plot()

    def configure_plot(self):
        """配置图表样式"""
        self.axes.set_title('Memory usage situation', fontsize=10)
        self.axes.set_xlabel('Time', fontsize=9)
        self.axes.set_ylabel('Memory (MB)', fontsize=9)
        self.axes.grid(True, linestyle='--', alpha=0.7)
        self.axes.tick_params(axis='both', which='major', labelsize=8)

        # 设置x轴日期格式
        self.axes.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
        self.axes.xaxis.set_major_locator(mdates.AutoDateLocator())

        # 紧凑布局
        self.figure.tight_layout()

    def update_chart(self, timestamps, memory_values):
        """更新图表数据"""
        self.timestamps = timestamps[-self.max_points:]
        self.memory_values = memory_values[-self.max_points:]

        # 清除当前图表
        self.axes.clear()

        # 重新配置图表
        self.configure_plot()

        # 绘制新数据
        if self.timestamps:
            self.axes.plot(self.timestamps, self.memory_values, 'b-', linewidth=1.5, marker='o', markersize=3)

            # 为最新点添加标签
            if len(self.timestamps) > 0:
                latest_time = self.timestamps[-1]
                latest_value = self.memory_values[-1]
                self.axes.annotate(f'{latest_value:.1f} MB',
                                   xy=(latest_time, latest_value),
                                   xytext=(10, 0), textcoords='offset points',
                                   arrowprops=dict(arrowstyle='->', color='red'),
                                   fontsize=8)

            # 自动调整y轴范围
            y_min = max(0, min(self.memory_values) * 0.9)
            y_max = max(self.memory_values) * 1.1
            if y_max == y_min:
                y_max = y_min + 10
            self.axes.set_ylim(y_min, y_max)

            # 自动旋转x轴标签
            self.figure.autofmt_xdate()

        # 刷新图表
        self.draw()


class ProcessMonitorUI(QMainWindow):
    """主窗口类"""

    def __init__(self):
        super().__init__()

        # 窗口设置
        self.setWindowTitle("进程内存监控工具")
        self.setGeometry(100, 100, 1000, 700)

        # 数据存储
        self.monitor_data = []
        self.monitoring = False

        # 创建UI
        self.init_ui()

        # 创建监控线程
        self.monitor_thread = None

    def init_ui(self):
        """初始化用户界面"""
        # 中央部件
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        # 主布局
        main_layout = QVBoxLayout(central_widget)

        # 顶部控制区域
        control_frame = QFrame()
        control_frame.setFrameShape(QFrame.StyledPanel)
        control_layout = QVBoxLayout(control_frame)

        # 标题
        title_label = QLabel("进程内存监控工具")
        title_label.setFont(QFont("Arial", 16, QFont.Bold))
        title_label.setAlignment(Qt.AlignCenter)
        control_layout.addWidget(title_label)

        # 进程选择区域
        process_layout = QHBoxLayout()

        # 选项：按名称或PID
        self.search_type_combo = QComboBox()
        self.search_type_combo.addItems(["按进程名", "按PID"])
        self.search_type_combo.currentIndexChanged.connect(self.toggle_search_type)
        process_layout.addWidget(QLabel("查找方式:"))
        process_layout.addWidget(self.search_type_combo, 1)

        # 进程名输入
        self.process_name_input = QLineEdit()
        self.process_name_input.setPlaceholderText("输入进程名(例如: chrome, python)")
        process_layout.addWidget(QLabel("进程名:"))
        process_layout.addWidget(self.process_name_input, 2)

        # PID输入
        self.pid_input = QLineEdit()
        self.pid_input.setPlaceholderText("输入进程PID")
        self.pid_input.setVisible(False)  # 默认隐藏
        process_layout.addWidget(QLabel("PID:"))
        process_layout.addWidget(self.pid_input, 2)

        control_layout.addLayout(process_layout)

        # 控制按钮
        button_layout = QHBoxLayout()

        self.start_btn = QPushButton("开始监控")
        self.start_btn.setStyleSheet("background-color: #4CAF50; color: white; font-weight: bold;")
        self.start_btn.clicked.connect(self.start_monitoring)

        self.stop_btn = QPushButton("停止监控")
        self.stop_btn.setStyleSheet("background-color: #f44336; color: white; font-weight: bold;")
        self.stop_btn.setEnabled(False)
        self.stop_btn.clicked.connect(self.stop_monitoring)

        self.save_btn = QPushButton("保存数据")
        self.save_btn.setEnabled(False)
        self.save_btn.clicked.connect(self.save_data)

        button_layout.addWidget(self.start_btn)
        button_layout.addWidget(self.stop_btn)
        button_layout.addWidget(self.save_btn)
        button_layout.addStretch()

        control_layout.addLayout(button_layout)

        # 间隔控制
        interval_layout = QHBoxLayout()
        interval_layout.addWidget(QLabel("监控间隔(秒):"))

        self.interval_input = QLineEdit("1.0")
        self.interval_input.setFixedWidth(80)
        interval_layout.addWidget(self.interval_input)
        interval_layout.addStretch()

        control_layout.addLayout(interval_layout)

        main_layout.addWidget(control_frame)

        # 图表区域
        chart_frame = QFrame()
        chart_frame.setFrameShape(QFrame.StyledPanel)
        chart_layout = QVBoxLayout(chart_frame)

        # 创建图表
        self.memory_chart = MemoryChart(self, width=5, height=4, dpi=100)
        chart_layout.addWidget(self.memory_chart)

        main_layout.addWidget(chart_frame, 1)

        # 信息显示区域
        info_frame = QFrame()
        info_frame.setFrameShape(QFrame.StyledPanel)
        info_layout = QHBoxLayout(info_frame)

        self.process_info_label = QLabel("未选择进程")
        self.process_info_label.setFont(QFont("Arial", 10))

        self.memory_info_label = QLabel("内存使用: -")
        self.memory_info_label.setFont(QFont("Arial", 10))

        self.status_label = QLabel("状态: 未监控")
        self.status_label.setFont(QFont("Arial", 10))

        info_layout.addWidget(self.process_info_label, 1)
        info_layout.addWidget(self.memory_info_label, 1)
        info_layout.addWidget(self.status_label, 1)

        main_layout.addWidget(info_frame)

        # 状态栏
        self.statusBar = QStatusBar()
        self.setStatusBar(self.statusBar)
        self.statusBar.showMessage("就绪")

    def toggle_search_type(self, index):
        """切换搜索类型（进程名或PID）"""
        if index == 0:  # 按进程名
            self.process_name_input.setVisible(True)
            self.pid_input.setVisible(False)
        else:  # 按PID
            self.process_name_input.setVisible(False)
            self.pid_input.setVisible(True)

    def start_monitoring(self):
        """开始监控进程"""
        if self.monitoring:
            return

        # 获取监控参数
        search_type = self.search_type_combo.currentIndex()
        interval_text = self.interval_input.text()

        try:
            interval = float(interval_text)
            if interval <= 0:
                raise ValueError("间隔必须大于0")
        except ValueError:
            QMessageBox.warning(self, "输入错误", "请输入有效的监控间隔（正数）")
            return

        # 创建并启动监控线程
        if search_type == 0:  # 按进程名
            process_name = self.process_name_input.text().strip()
            if not process_name:
                QMessageBox.warning(self, "输入错误", "请输入进程名")
                return
            self.monitor_thread = ProcessMonitorThread(process_name=process_name, interval=interval)
        else:  # 按PID
            pid_text = self.pid_input.text().strip()
            if not pid_text or not pid_text.isdigit():
                QMessageBox.warning(self, "输入错误", "请输入有效的PID")
                return
            self.monitor_thread = ProcessMonitorThread(pid=int(pid_text), interval=interval)

        # 连接信号
        self.monitor_thread.data_updated.connect(self.update_data)
        self.monitor_thread.process_ended.connect(self.handle_process_ended)
        self.monitor_thread.error_occurred.connect(self.handle_error)

        # 更新UI状态
        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.save_btn.setEnabled(False)
        self.status_label.setText("状态: 监控中...")
        self.status_label.setStyleSheet("color: green;")
        self.monitoring = True
        self.monitor_data = []  # 重置数据

        # 启动线程
        self.monitor_thread.start()
        self.statusBar.showMessage(f"开始监控 {self.process_name_input.text() or self.pid_input.text()}")

    def stop_monitoring(self):
        """停止监控进程"""
        if self.monitor_thread and self.monitoring:
            self.monitor_thread.stop()
            self.monitoring = False

        # 更新UI状态
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.save_btn.setEnabled(len(self.monitor_data) > 0)
        self.status_label.setText("状态: 已停止")
        self.status_label.setStyleSheet("color: orange;")
        self.statusBar.showMessage("监控已停止")

    def update_data(self, data):
        """更新图表和显示信息"""
        # 保存数据
        self.monitor_data.append(data)

        # 更新图表
        timestamps = [d['timestamp'] for d in self.monitor_data]
        memory_values = [d['memory_rss'] for d in self.monitor_data]
        self.memory_chart.update_chart(timestamps, memory_values)

        # 更新信息标签
        self.process_info_label.setText(f"进程: {data['name']} (PID: {data['pid']})")
        self.memory_info_label.setText(
            f"内存: {data['memory_rss']:.1f} MB ({data['memory_percent']:.1f}%) | "
            f"CPU: {data['cpu_percent']:.1f}%"
        )

        # 启用保存按钮
        self.save_btn.setEnabled(True)

    def handle_process_ended(self):
        """处理进程结束事件"""
        self.stop_monitoring()
        QMessageBox.information(self, "进程结束", "被监控的进程已结束运行")
        self.statusBar.showMessage("被监控的进程已结束")

    def handle_error(self, error_message):
        """处理错误"""
        self.stop_monitoring()
        QMessageBox.critical(self, "错误", error_message)
        self.statusBar.showMessage(f"错误: {error_message}")

    def save_data(self):
        """保存监控数据到CSV文件"""
        if not self.monitor_data:
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
            # 保存为CSV
            df = pd.DataFrame(self.monitor_data)
            # 格式化时间戳
            df['timestamp'] = df['timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S.%f')
            df.to_csv(file_path, index=False, encoding='utf-8-sig')

            QMessageBox.information(self, "保存成功", f"监控数据已保存到:\n{file_path}")
            self.statusBar.showMessage(f"数据已保存至 {file_path}")
        except Exception as e:
            QMessageBox.critical(self, "保存失败", f"保存数据时出错:\n{str(e)}")
            self.statusBar.showMessage(f"保存失败: {str(e)}")

    def closeEvent(self, event):
        """处理窗口关闭事件"""
        self.stop_monitoring()
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

    # 创建并显示主窗口
    window = ProcessMonitorUI()
    window.show()

    # 运行应用
    sys.exit(app.exec_())
