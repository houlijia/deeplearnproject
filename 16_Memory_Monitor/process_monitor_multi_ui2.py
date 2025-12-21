import sys
import psutil
import pandas as pd
from datetime import datetime
from PyQt5.QtWidgets import *
from PyQt5.QtCore import QTimer
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure


class ProcessMonitorApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("多进程监控工具")
        self.setGeometry(100, 100, 1000, 700)

        # 初始化数据结构
        self.process_data = {}  # {pid: {"name": str, "memory": list, "timestamps": list}}
        self.csv_files = {}

        # 创建UI
        self.create_ui()

        # 定时器
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_data)
        self.monitoring = False

    def create_ui(self):
        # 主布局
        main_widget = QWidget()
        main_layout = QVBoxLayout()

        # 输入区域
        input_layout = QHBoxLayout()
        self.pid_input = QLineEdit()
        self.pid_input.setPlaceholderText("输入PID（多个用逗号分隔）")
        add_button = QPushButton("添加进程")
        add_button.clicked.connect(self.add_processes)
        start_button = QPushButton("开始监控")
        start_button.clicked.connect(self.start_monitoring)
        stop_button = QPushButton("停止监控")
        stop_button.clicked.connect(self.stop_monitoring)

        input_layout.addWidget(QLabel("监控进程:"))
        input_layout.addWidget(self.pid_input)
        input_layout.addWidget(add_button)
        input_layout.addWidget(start_button)
        input_layout.addWidget(stop_button)

        # 进程列表
        self.process_list = QListWidget()

        # 图表区域
        self.figure = Figure(figsize=(10, 6))
        self.canvas = FigureCanvas(self.figure)
        self.ax = self.figure.add_subplot(111)
        self.ax.set_title("Memory usage situation")
        self.ax.set_xlabel("Time")
        self.ax.set_ylabel("Memory (MB)")
        self.ax.grid(True)

        # 添加到主布局
        main_layout.addLayout(input_layout)
        main_layout.addWidget(QLabel("监控列表:"))
        main_layout.addWidget(self.process_list)
        main_layout.addWidget(self.canvas)

        main_widget.setLayout(main_layout)
        self.setCentralWidget(main_widget)

    def add_processes(self):
        """添加要监控的进程"""
        pids = self.pid_input.text().split(',')
        for pid_str in pids:
            try:
                pid = int(pid_str.strip())
                if pid in self.process_data:
                    continue

                # 检查进程是否存在
                p = psutil.Process(pid)
                name = p.name()

                # 初始化数据结构
                self.process_data[pid] = {
                    "name": name,
                    "memory": [],
                    "timestamps": []
                }

                # 创建CSV文件
                filename = f"process_{pid}_{name}.csv"
                self.csv_files[pid] = filename
                with open(filename, 'w') as f:
                    f.write("Timestamp,Memory(MB)\n")

                # 添加到UI列表
                self.process_list.addItem(f"{pid} - {name}")

            except (psutil.NoSuchProcess, ValueError):
                QMessageBox.warning(self, "错误", f"无效的PID: {pid_str}")

    def start_monitoring(self):
        """开始监控进程"""
        if not self.process_data:
            QMessageBox.warning(self, "错误", "请先添加要监控的进程")
            return

        self.monitoring = True
        self.timer.start(1000)  # 每秒更新一次

    def stop_monitoring(self):
        """停止监控"""
        self.monitoring = False
        self.timer.stop()

    def update_data(self):
        """更新监控数据并刷新图表"""
        if not self.monitoring:
            return

        # 清空图表
        self.ax.clear()
        self.ax.set_title("Memory usage situation")
        self.ax.set_xlabel("Time")
        self.ax.set_ylabel("Memory (MB)")
        self.ax.grid(True)

        # 收集当前时间
        current_time = datetime.now().strftime("%H:%M:%S")

        # 更新每个进程的数据
        for pid, data in self.process_data.items():
            try:
                p = psutil.Process(pid)
                mem_info = p.memory_info()
                mem_mb = round(mem_info.rss / (1024 * 1024), 2)  # 转换为MB

                # 更新数据
                data["memory"].append(mem_mb)
                data["timestamps"].append(current_time)

                # 写入CSV
                with open(self.csv_files[pid], 'a') as f:
                    f.write(f"{current_time},{mem_mb}\n")

                # 绘制折线图
                self.ax.plot(
                    data["timestamps"],
                    data["memory"],
                    label=f"{pid} - {data['name']}"
                )

            except psutil.NoSuchProcess:
                # 进程已结束
                self.process_data.pop(pid, None)
                self.csv_files.pop(pid, None)
                # 从列表中移除
                for i in range(self.process_list.count()):
                    if str(pid) in self.process_list.item(i).text():
                        self.process_list.takeItem(i)
                        break

        # 添加图例和美化
        self.ax.legend(loc='upper left')
        self.ax.tick_params(axis='x', rotation=45)
        self.canvas.draw()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = ProcessMonitorApp()
    window.show()
    sys.exit(app.exec_())
