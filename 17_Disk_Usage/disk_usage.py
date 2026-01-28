import os
import sys
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QComboBox, QProgressBar, QTextEdit,
    QFileDialog, QMessageBox, QGroupBox, QSpinBox
)
from PyQt5.QtCore import QThread, pyqtSignal, Qt
import shutil


class DiskFillerThread(QThread):
    progress_signal = pyqtSignal(int)  # 发送进度百分比
    log_signal = pyqtSignal(str)  # 发送日志信息
    finished_signal = pyqtSignal()  # 发送完成信号

    def __init__(self, target_path, target_percentage, fill_filename_prefix='FILLER_'):
        super().__init__()
        self.target_path = target_path
        self.target_percentage = target_percentage
        self.fill_filename_prefix = fill_filename_prefix
        self._is_stopped = False

    def stop(self):
        self._is_stopped = True

    def run(self):
        try:
            # 1. 获取磁盘总空间和可用空间
            total_space = shutil.disk_usage(self.target_path).total
            free_space = shutil.disk_usage(self.target_path).free
            used_space = total_space - free_space

            # 计算目标占用空间（字节）
            target_used_bytes = int(total_space * (self.target_percentage / 100.0))

            # 如果当前已用空间已经大于等于目标，则提示
            if used_space >= target_used_bytes:
                self.log_signal.emit(
                    f"当前已用空间 ({used_space / (1024 ** 3):.2f} GB) 已达到或超过目标 ({target_used_bytes / (1024 ** 3):.2f} GB)。")
                self.finished_signal.emit()
                return

            # 计算还需要多少空间
            bytes_to_fill = target_used_bytes - used_space
            self.log_signal.emit(f"开始占用 {bytes_to_fill / (1024 ** 3):.2f} GB 空间...")

            # 2. 创建填充文件
            fill_file_path = os.path.join(self.target_path, f"{self.fill_filename_prefix}auto_generated.dat")
            chunk_size = 1024 * 1024 * 10  # 10MB chunks to avoid memory issues
            filled_so_far = 0

            with open(fill_file_path, 'wb') as f:
                while filled_so_far < bytes_to_fill and not self._is_stopped:
                    # Calculate how much to write in this iteration
                    current_chunk_size = min(chunk_size, bytes_to_fill - filled_so_far)
                    f.write(b'\0' * current_chunk_size)
                    filled_so_far += current_chunk_size

                    # Update progress
                    current_total_used = used_space + filled_so_far
                    progress_percent = int((current_total_used / total_space) * 100)
                    self.progress_signal.emit(progress_percent)

            # Check if stopped by user
            if self._is_stopped:
                self.log_signal.emit("用户请求停止，正在中断...")
                # Attempt to clean up the partially written file
                if os.path.exists(fill_file_path):
                    try:
                        os.remove(fill_file_path)
                        self.log_signal.emit("已清理未完成的填充文件。")
                    except OSError as e:
                        self.log_signal.emit(f"清理未完成文件失败: {e}")
            else:
                self.log_signal.emit(f"填充完成！目标占用率 {self.target_percentage}% 已达成。")

        except Exception as e:
            self.log_signal.emit(f"发生错误: {str(e)}")
        finally:
            self.finished_signal.emit()


class DiskFillerGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('磁盘空间占用工具')
        self.setGeometry(100, 100, 600, 500)

        # Central Widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)

        # --- Drive Selection ---
        drive_group = QGroupBox("选择磁盘")
        drive_layout = QHBoxLayout()

        self.drive_combo = QComboBox()
        self.refresh_drives()

        self.refresh_button = QPushButton("刷新")
        self.refresh_button.clicked.connect(self.refresh_drives)

        drive_layout.addWidget(self.drive_combo)
        drive_layout.addWidget(self.refresh_button)
        drive_group.setLayout(drive_layout)
        layout.addWidget(drive_group)

        # --- Fill Settings ---
        settings_group = QGroupBox("占用设置")
        settings_layout = QVBoxLayout()

        self.percentage_label = QLabel("占用百分比:")
        self.percentage_input = QSpinBox()
        self.percentage_input.setRange(1, 99)
        self.percentage_input.setValue(50)

        percentage_hbox = QHBoxLayout()
        percentage_hbox.addWidget(self.percentage_label)
        percentage_hbox.addWidget(self.percentage_input)
        percentage_hbox.addStretch()

        settings_layout.addLayout(percentage_hbox)
        settings_group.setLayout(settings_layout)
        layout.addWidget(settings_group)

        # --- Progress Bar ---
        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(0)
        layout.addWidget(self.progress_bar)

        # --- Buttons ---
        button_layout = QHBoxLayout()
        self.start_button = QPushButton("开始占用")
        self.stop_button = QPushButton("停止占用")
        self.stop_button.setEnabled(False)  # Initially disabled
        self.clean_button = QPushButton("清理占用文件")

        self.start_button.clicked.connect(self.start_filling)
        self.stop_button.clicked.connect(self.stop_filling)
        self.clean_button.clicked.connect(self.clean_files)

        button_layout.addWidget(self.start_button)
        button_layout.addWidget(self.stop_button)
        button_layout.addWidget(self.clean_button)
        layout.addLayout(button_layout)

        # --- Log Output ---
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        layout.addWidget(self.log_text)

        # --- State Variables ---
        self.fill_thread = None

    def refresh_drives(self):
        """Refresh the list of available drives."""
        self.drive_combo.clear()
        # This works on both Windows and Unix-like systems
        # For more robust drive detection, consider using psutil
        if os.name == 'nt':  # Windows
            import string
            from ctypes import windll
            bitmask = windll.kernel32.GetLogicalDrives()
            for letter in string.ascii_uppercase:
                if bitmask & 1:
                    path = f"{letter}:\\"
                    if os.path.ismount(path):  # Check if it's a mount point (drive)
                        self.drive_combo.addItem(path)
                bitmask >>= 1
        else:  # POSIX (Linux, macOS)
            # A common approach is to list root and subdirs like /mnt, /media
            # For simplicity here, we'll just add the root
            # A more advanced method would parse /proc/mounts or use psutil
            self.drive_combo.addItem("/")
            # Example with psutil: uncomment and install psutil if needed
            # import psutil
            # partitions = psutil.disk_partitions(all=False)
            # for partition in partitions:
            #     self.drive_combo.addItem(partition.mountpoint)

    def start_filling(self):
        selected_drive = self.drive_combo.currentText()
        if not selected_drive:
            QMessageBox.warning(self, "警告", "请选择一个磁盘驱动器或路径。")
            return

        target_percentage = self.percentage_input.value()
        if target_percentage <= 0 or target_percentage > 99:
            QMessageBox.warning(self, "警告", "请输入一个有效的百分比 (1-99)。")
            return

        # Disable UI elements during operation
        self.start_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        self.drive_combo.setEnabled(False)
        self.percentage_input.setEnabled(False)
        self.log_text.append("--- 开始占用磁盘空间 ---")

        # Create and start the thread
        self.fill_thread = DiskFillerThread(selected_drive, target_percentage)
        self.fill_thread.progress_signal.connect(self.update_progress)
        self.fill_thread.log_signal.connect(self.append_log)
        self.fill_thread.finished_signal.connect(self.on_fill_finished)
        self.fill_thread.start()

    def stop_filling(self):
        if self.fill_thread and self.fill_thread.isRunning():
            self.fill_thread.stop()
            self.append_log("正在请求停止...")
            # Note: The thread will finish its current write block before stopping cleanly.

    def update_progress(self, value):
        self.progress_bar.setValue(value)

    def append_log(self, message):
        self.log_text.append(message)

    def on_fill_finished(self):
        # Re-enable UI elements
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self.drive_combo.setEnabled(True)
        self.percentage_input.setEnabled(True)
        self.append_log("--- 占用操作结束 ---")

    def clean_files(self):
        selected_drive = self.drive_combo.currentText()
        if not selected_drive:
            QMessageBox.warning(self, "警告", "请选择一个磁盘驱动器或路径。")
            return

        # Look for files starting with the prefix used by the filler
        prefix = "FILLER_"
        found_files = []
        for f in os.listdir(selected_drive):
            if f.startswith(prefix) and f.endswith('.dat'):
                full_path = os.path.join(selected_drive, f)
                if os.path.isfile(full_path):
                    found_files.append(full_path)

        if not found_files:
            QMessageBox.information(self, "信息", f"在 {selected_drive} 中未找到由本工具生成的占用文件。")
            return

        reply = QMessageBox.question(
            self, '确认清理',
            f"找到 {len(found_files)} 个占用文件，确定要删除它们吗？\n{', '.join(found_files[:3])}...",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No
        )

        if reply == QMessageBox.Yes:
            errors_occurred = False
            for file_path in found_files:
                try:
                    os.remove(file_path)
                    self.append_log(f"已删除: {file_path}")
                except OSError as e:
                    self.append_log(f"删除失败 {file_path}: {e}")
                    errors_occurred = True
            if not errors_occurred:
                self.append_log("所有占用文件均已成功清理。")
            else:
                self.append_log("部分文件清理失败，请检查权限或手动删除。")


def main():
    app = QApplication(sys.argv)
    window = DiskFillerGUI()
    window.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()