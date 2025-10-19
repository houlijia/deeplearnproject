import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import os
import shutil
from datetime import datetime


class FileProcessorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Xmind2Excel")
        self.root.geometry("700x500")
        self.root.resizable(True, True)

        # 创建样式
        self.style = ttk.Style()
        self.style.configure("TButton", font=("Arial", 10), padding=6)
        self.style.map("TButton",
                       background=[("active", "#3498db")],
                       foreground=[("active", "white")]
                       )

        # 初始化变量
        self.selected_file = ""
        self.converted_file = ""
        self.operations = ["xmind转为excel"]

        self.setup_ui()

    def setup_ui(self):
        # 主框架
        main_frame = ttk.Frame(self.root, padding=20)
        main_frame.pack(fill=tk.BOTH, expand=True)

        # 标题
        title_label = ttk.Label(
            main_frame,
            text="欢迎使用Xmind To Excel Tool",
            font=("Arial", 18, "bold"),
            foreground="#2c3e50"
        )
        title_label.grid(row=0, column=0, columnspan=2, pady=(0, 20))

        # 文件选择区域
        file_frame = ttk.LabelFrame(main_frame, text="文件操作", padding=10)
        file_frame.grid(row=1, column=0, padx=5, pady=5, sticky="ew")

        ttk.Button(
            file_frame,
            text="上传文件",
            command=self.upload_file,
            style="TButton"
        ).grid(row=0, column=0, padx=5, pady=5)

        self.file_label = ttk.Label(
            file_frame,
            text="未选择文件",
            foreground="#7f8c8d"
        )
        self.file_label.grid(row=0, column=1, padx=5, pady=5, sticky="w")

        # 文件转换区域
        convert_frame = ttk.LabelFrame(main_frame, text="文件转换", padding=10)
        convert_frame.grid(row=2, column=0, padx=5, pady=5, sticky="ew")

        ttk.Label(convert_frame, text="选择操作:").grid(row=0, column=0, padx=5, pady=5, sticky="w")

        self.operation_var = tk.StringVar()
        self.operation_var.set(self.operations[0])

        operation_menu = ttk.Combobox(
            convert_frame,
            textvariable=self.operation_var,
            values=self.operations,
            state="readonly",
            width=15
        )
        operation_menu.grid(row=0, column=1, padx=5, pady=5, sticky="w")

        ttk.Button(
            convert_frame,
            text="执行转换",
            command=self.convert_file,
            style="TButton"
        ).grid(row=0, column=2, padx=5, pady=5)

        # 下载区域
        download_frame = ttk.LabelFrame(main_frame, text="下载文件", padding=10)
        download_frame.grid(row=3, column=0, padx=5, pady=5, sticky="ew")

        self.download_label = ttk.Label(
            download_frame,
            text="无可用文件下载",
            foreground="#7f8c8d"
        )
        self.download_label.grid(row=0, column=0, padx=5, pady=5, sticky="w")

        ttk.Button(
            download_frame,
            text="下载文件",
            command=self.download_file,
            style="TButton"
        ).grid(row=0, column=1, padx=5, pady=5)

        # 文件信息区域
        info_frame = ttk.LabelFrame(main_frame, text="文件信息", padding=10)
        info_frame.grid(row=1, column=1, rowspan=3, padx=5, pady=5, sticky="nsew")

        self.info_text = tk.Text(
            info_frame,
            height=15,
            width=30,
            state=tk.DISABLED
        )
        self.info_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # 状态栏
        self.status_var = tk.StringVar()
        self.status_var.set("就绪")
        status_bar = ttk.Label(
            self.root,
            textvariable=self.status_var,
            relief=tk.SUNKEN,
            anchor=tk.W
        )
        status_bar.pack(side=tk.BOTTOM, fill=tk.X)

        # 配置网格权重
        main_frame.columnconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(4, weight=1)

    def upload_file(self):
        """上传文件处理函数"""
        file_path = filedialog.askopenfilename(
            title="选择文件",
            filetypes=[
                ("文本文件", "*.txt"),
                ("所有文件", "*.*")
            ]
        )

        if file_path:
            self.selected_file = file_path
            filename = os.path.basename(file_path)
            self.file_label.config(text=filename, foreground="#27ae60")
            self.status_var.set(f"已选择文件: {filename}")
            self.update_file_info()

    def convert_file(self):
        """文件转换处理函数"""
        if not self.selected_file:
            messagebox.showerror("错误", "请先上传文件")
            return

        operation = self.operation_var.get()
        self.status_var.set(f"正在执行操作: {operation}...")
        self.root.update()

        try:
            # 读取文件内容
            with open(self.selected_file, 'r', encoding='utf-8') as f:
                content = f.read()

            # 执行转换操作
            if operation == "xmind转为excel":
                converted_content = content.upper()
            elif operation == "转为小写":
                converted_content = content.lower()
            elif operation == "反转内容":
                converted_content = content[::-1]
            elif operation == "添加时间戳":
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                converted_content = f"[{timestamp}]\n{content}"

            # 保存转换后的文件
            filename, ext = os.path.splitext(os.path.basename(self.selected_file))
            converted_filename = f"{filename}_{operation}{ext}"
            self.converted_file = os.path.join(os.path.dirname(self.selected_file), converted_filename)

            with open(self.converted_file, 'w', encoding='utf-8') as f:
                f.write(converted_content)

            self.download_label.config(text=converted_filename, foreground="#27ae60")
            self.status_var.set(f"已完成转换: {operation}")
            self.update_file_info()

        except Exception as e:
            messagebox.showerror("转换错误", f"转换过程中出错: {str(e)}")
            self.status_var.set("转换失败")

    def download_file(self):
        """下载文件处理函数"""
        if not self.converted_file:
            messagebox.showerror("错误", "没有可下载的文件")
            return

        download_path = filedialog.asksaveasfilename(
            title="保存文件",
            initialfile=os.path.basename(self.converted_file),
            filetypes=[
                ("文本文件", "*.txt"),
                ("所有文件", "*.*")
            ]
        )

        if download_path:
            try:
                shutil.copy(self.converted_file, download_path)
                messagebox.showinfo("成功", f"文件已保存到:\n{download_path}")
                self.status_var.set(f"文件下载完成: {os.path.basename(download_path)}")
            except Exception as e:
                messagebox.showerror("下载错误", f"下载过程中出错: {str(e)}")
                self.status_var.set("下载失败")

    def update_file_info(self):
        """更新文件信息区域"""
        self.info_text.config(state=tk.NORMAL)
        self.info_text.delete(1.0, tk.END)

        if self.selected_file:
            # 获取文件信息
            file_stats = os.stat(self.selected_file)
            size_kb = file_stats.st_size / 1024
            created = datetime.fromtimestamp(file_stats.st_ctime).strftime("%Y-%m-%d %H:%M:%S")
            modified = datetime.fromtimestamp(file_stats.st_mtime).strftime("%Y-%m-%d %H:%M:%S")

            info = f"原始文件: {os.path.basename(self.selected_file)}\n"
            info += f"大小: {size_kb:.2f} KB\n"
            info += f"创建时间: {created}\n"
            info += f"修改时间: {modified}\n"
            info += "=" * 30 + "\n"

            if self.converted_file:
                conv_stats = os.stat(self.converted_file)
                conv_size_kb = conv_stats.st_size / 1024
                conv_modified = datetime.fromtimestamp(conv_stats.st_mtime).strftime("%Y-%m-%d %H:%M:%S")

                info += f"转换后文件: {os.path.basename(self.converted_file)}\n"
                info += f"大小: {conv_size_kb:.2f} KB\n"
                info += f"转换时间: {conv_modified}\n"
                info += f"操作类型: {self.operation_var.get()}"

                # 添加文件预览（前10行）
                try:
                    with open(self.converted_file, 'r', encoding='utf-8') as f:
                        preview = f.readlines()[:10]
                    info += "\n\n预览:\n" + "".join(preview)
                except:
                    info += "\n\n(无法预览文件内容)"

            self.info_text.insert(tk.END, info)

        self.info_text.config(state=tk.DISABLED)


if __name__ == "__main__":
    root = tk.Tk()
    app = FileProcessorApp(root)
    root.mainloop()
