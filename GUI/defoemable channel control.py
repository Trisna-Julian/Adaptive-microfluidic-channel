import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox
import serial
import serial.tools.list_ports
import threading
import cv2
from PIL import Image, ImageTk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import nidaqmx
from nidaqmx.constants import AcquisitionType
import numpy as np
import time
from scipy.signal import butter, filtfilt

import os

from collections import deque
import datetime



class SerialMonitor:
    def __init__(self, root):
        self.root = root
        self.root.title("Deformable channel controller")

        self.serial_port = None
        self.running = False
        self.canny_mode = False

        self.canny_thresh1 = tk.IntVar(value=0)
        self.canny_thresh2 = tk.IntVar(value=60)
        
        # Store the last N predictions
        self.prediction_buffer = deque(maxlen=5)
        # Initialize time tracker
        self.last_pred_time = 0
        self.prediction_interval = 0.5  # seconds
        self.current_prediction = None  # store last prediction

        self.approaching = False
        self.deaproaching = False
        self.guard = False
        self.releasing = False

        self.current_frame = 0

        self.create_widgets()
        self.start_camera()
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

        


    def create_widgets(self):
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        self.left_frame = ttk.Frame(main_frame)
        self.left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5)

        self.right_frame = ttk.Frame(main_frame)
        self.right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5)

        self.build_serial_controls()
        self.build_serial_monitor()
        self.build_camera_display()

    def build_serial_controls(self):
        top_frame = ttk.Frame(self.left_frame)
        top_frame.pack(pady=5)

        ttk.Label(top_frame, text="Port:").grid(row=0, column=0)
        self.port_cb = ttk.Combobox(top_frame, values=self.get_serial_ports(), width=10)
        self.port_cb.set("COM3")
        self.port_cb.grid(row=0, column=1)

        ttk.Label(top_frame, text="Baud:").grid(row=0, column=2)
        self.baud_cb = ttk.Combobox(top_frame, values=["115200"], width=10)
        self.baud_cb.set("115200")
        self.baud_cb.grid(row=0, column=3)

        self.connect_btn = ttk.Button(top_frame, text="Connect", command=self.toggle_connection)
        self.connect_btn.grid(row=0, column=4, padx=10)

        bottom_frame = ttk.Frame(self.left_frame)
        bottom_frame.pack()

        self.entry = ttk.Entry(bottom_frame, width=40)
        self.entry.grid(row=0, column=0, padx=5)
        self.entry.bind("<Return>", lambda event: self.send_data())

        self.send_btn = ttk.Button(bottom_frame, text="Send", command=self.send_data)
        self.send_btn.grid(row=0, column=1)

        command_frame = ttk.Frame(self.left_frame)
        command_frame.pack(pady=5)

        ttk.Button(command_frame, text="m100;", command=lambda: self.send_command("m100;")).pack(side=tk.LEFT, padx=5)
        ttk.Button(command_frame, text="m-100;", command=lambda: self.send_command("m-100;")).pack(side=tk.LEFT, padx=5)
        ttk.Button(command_frame, text="Home;", command=lambda: self.home).pack(side=tk.LEFT, padx=5)
        ttk.Button(command_frame, text="p0", command=lambda: self.send_command("p0")).pack(side=tk.LEFT, padx=5)
        ttk.Button(command_frame, text="Approach", command=lambda: self.approach()).pack(side=tk.LEFT, padx=5)
        ttk.Button(command_frame, text="Guard", command=lambda: self.guard_activate()).pack(side=tk.LEFT, padx=5)

        result_frame = ttk.Frame(self.left_frame)
        result_frame.pack(pady=5)

        self.approach_text = tk.StringVar(value="Ready")
        self.approach_label = tk.Label(result_frame, textvariable=self.approach_text, font=("Helvetica", 16), fg="blue")
        self.approach_label.pack(side=tk.LEFT, padx=5)

        # Result label
        self.result_text = tk.StringVar()
        self.result_label = tk.Label(result_frame, textvariable=self.result_text, font=("Helvetica", 16), fg="blue")
        self.result_label.pack(side=tk.LEFT, padx=5)


    def build_serial_monitor(self):
        self.output = scrolledtext.ScrolledText(self.left_frame, wrap=tk.WORD, height=20, width=50)
        self.output.pack(padx=5, pady=10)

    def build_camera_display(self):
        self.notebook = ttk.Notebook(self.right_frame)
        self.notebook.pack(pady=5, fill=tk.BOTH, expand=True)

        self.negative_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.negative_tab, text="Negative View")

        self.canny_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.canny_tab, text="Canny Edge View")

        self.plot_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.plot_tab, text="ADC Plot")
        self.setup_adc_plot()

        self.negative_label = ttk.Label(self.negative_tab)
        self.negative_label.pack(pady=5)

        slider_frame = ttk.Frame(self.canny_tab)
        slider_frame.pack(pady=10)

        t1_frame = ttk.Frame(slider_frame)
        t1_frame.pack(side=tk.LEFT, padx=20)

        ttk.Label(t1_frame, text="Threshold 1:").grid(row=0, column=0, sticky="w")
        t1_scale = ttk.Scale(t1_frame, from_=0, to=255, variable=self.canny_thresh1,
                            orient=tk.HORIZONTAL, length=150, command=lambda e: self.update_threshold_labels())
        t1_scale.grid(row=0, column=1, padx=5)
        self.t1_label = ttk.Label(t1_frame, text=str(self.canny_thresh1.get()))
        self.t1_label.grid(row=0, column=2)

        t2_frame = ttk.Frame(slider_frame)
        t2_frame.pack(side=tk.LEFT, padx=20)

        ttk.Label(t2_frame, text="Threshold 2:").grid(row=0, column=0, sticky="w")
        t2_scale = ttk.Scale(t2_frame, from_=0, to=255, variable=self.canny_thresh2,
                            orient=tk.HORIZONTAL, length=150, command=lambda e: self.update_threshold_labels())
        t2_scale.grid(row=0, column=1, padx=5)
        self.t2_label = ttk.Label(t2_frame, text=str(self.canny_thresh2.get()))
        self.t2_label.grid(row=0, column=2)

        edge_frame = ttk.Frame(slider_frame)
        edge_frame.pack(side=tk.LEFT, padx=20)

        ttk.Label(edge_frame, text="Edges:").grid(row=0, column=0, sticky="w")
        self.edge_count_label = ttk.Label(edge_frame, text="0")
        self.edge_count_label.grid(row=0, column=1)

        self.canny_label = ttk.Label(self.canny_tab)
        self.canny_label.pack(pady=5)

        # Save button
        save_button = tk.Button(self.canny_tab, text="Save Image", command=self.save_image)
        save_button.pack(pady=5)

    def setup_adc_plot(self):
        self.auto_adjust = tk.BooleanVar(value=False)
        self.target_min_mv = tk.IntVar(value=150)
        self.target_max_mv = tk.IntVar(value=200)
        self.fig, self.ax = plt.subplots(figsize=(5, 3))
        self.line, = self.ax.plot([], [], lw=2)
        self.ax.set_title("ADC Channel - ai4 (Time in seconds)")
        self.ax.set_xlim(0, 1)
        self.ax.set_ylim(-1, 1)
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.plot_tab)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self.peak_label = ttk.Label(self.plot_tab, text="Average Peak Value: -- V")
        control_frame = ttk.Frame(self.plot_tab)
        control_frame.pack(pady=5)

        self.adjust_btn = ttk.Checkbutton(control_frame, text="Auto Adjust Peak", variable=self.auto_adjust)
        self.adjust_btn.grid(row=0, column=0, padx=5)

        ttk.Label(control_frame, text="Target Min (mV):").grid(row=0, column=1)
        self.min_value_label = ttk.Label(control_frame, textvariable=self.target_min_mv)
        self.min_value_label.grid(row=1, column=1)
        tk.Scale(control_frame, from_=0, to=1000, variable=self.target_min_mv, orient=tk.HORIZONTAL, length=150, resolution=10).grid(row=0, column=2)
        self.min_scale_value = ttk.Label(control_frame, textvariable=self.target_min_mv)
        self.min_scale_value.grid(row=1, column=2)

        ttk.Label(control_frame, text="Target Max (mV):").grid(row=0, column=3)
        self.max_value_label = ttk.Label(control_frame, textvariable=self.target_max_mv)
        self.max_value_label.grid(row=1, column=3)
        tk.Scale(control_frame, from_=0, to=1000, variable=self.target_max_mv, orient=tk.HORIZONTAL, length=150, resolution=10).grid(row=0, column=4)
        self.max_scale_value = ttk.Label(control_frame, textvariable=self.target_max_mv)
        self.max_scale_value.grid(row=1, column=4)
        self.peak_label.pack(pady=5)

        # self.root.after(1000, self.update_adc_plot)

    def high_pass_filter(self, data, cutoff=1, fs=125000, order=1):
        b, a = butter(order, cutoff / (0.5 * fs), btype='high', analog=False)
        return filtfilt(b, a, data)

    def calculate_average_peak(self, data, threshold=0.1):
        peaks = data[(np.roll(data, 1) < data) & (np.roll(data, -1) < data)]
        significant_peaks = peaks[peaks > threshold]
        if significant_peaks.size > 0:
            return np.mean(significant_peaks)
        else:
            return 0

    def update_adc_plot(self):
        TARGET_MIN = self.target_min_mv.get() / 1000
        TARGET_MAX = self.target_max_mv.get() / 1000
        STEP_SIZE = 10
        try:
            with nidaqmx.Task() as task:
                task.ai_channels.add_ai_voltage_chan("Dev2/ai4", min_val=0, max_val=3)
                task.timing.cfg_samp_clk_timing(rate=125000,
                                                sample_mode=AcquisitionType.FINITE,
                                                samps_per_chan=125000)
                data = task.read(number_of_samples_per_channel=125000)
                data = np.array(data)
                data = self.high_pass_filter(data)
                x = np.arange(125000) / 125000
                self.line.set_data(x, data)
                avg_peak = self.calculate_average_peak(data)
                if self.auto_adjust.get():
                    if avg_peak < TARGET_MIN:
                        self.send_command(f"m{STEP_SIZE}")
                    elif avg_peak > TARGET_MAX:
                        self.send_command(f"m{-STEP_SIZE}")
                self.peak_label.config(text=f"Average Peak Value: {avg_peak * 1000:.1f} mV")
                self.ax.set_xlim(0, 125000)
                self.canvas.draw()
        except Exception as e:
            print("ADC Error:", e)
        self.root.after(1000, self.update_adc_plot)

    def get_serial_ports(self):
        ports = serial.tools.list_ports.comports()
        return [port.device for port in ports]
    def toggle_connection(self):

        if self.serial_port and self.serial_port.is_open:
            self.running = False
            self.serial_port.close()
            self.connect_btn.config(text="Connect")
            self.output.insert(tk.END, "Disconnected\n")
        else:
            try:
                port = self.port_cb.get()
                baud = int(self.baud_cb.get())
                self.serial_port = serial.Serial(port, baud, timeout=1)
                self.running = True
                self.connect_btn.config(text="Disconnect")
                self.output.insert(tk.END, f"Connected to {port} @ {baud} baud\n")
                threading.Thread(target=self.read_serial, daemon=True).start()
            except Exception as e:
                messagebox.showerror("Connection Error", str(e))

    def read_serial(self):
        while self.running:
            if self.serial_port.in_waiting:
                data = self.serial_port.readline().decode('utf-8', errors='replace')
                self.output.insert(tk.END, data)
                self.output.see(tk.END)

    def send_data(self):
        if self.serial_port and self.serial_port.is_open:
            msg = self.entry.get() + '\n'
            self.serial_port.write(msg.encode())
            self.entry.delete(0, tk.END)
        else:
            messagebox.showwarning("Not Connected", "Please connect to a port first.")

    def send_command(self, command):
        if self.serial_port and self.serial_port.is_open:
            self.serial_port.write((command + '\n').encode())
            self.output.insert(tk.END, f"> {command}\n")
            self.output.see(tk.END)
        else:
            messagebox.showwarning("Not Connected", "Please connect to a port first.")

    def start_camera(self):
        self.cap = cv2.VideoCapture(0)
        self.running = True
        self.update_camera()

    def update_camera(self):
        ret, frame = self.cap.read()
        if ret:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            negative = cv2.bitwise_not(gray)
            negative_display = cv2.cvtColor(negative, cv2.COLOR_GRAY2RGB)
            height, width = gray.shape
            center_x, center_y = width // 2, height // 2
            top_left = (center_x - 50, center_y - 50)
            bottom_right = (center_x + 50, center_y + 50)

            height, width = gray.shape[:2]
            center_x, center_y = width // 2, height // 2

            roi_w, roi_h = (60,100)
            x1 = max(center_x - roi_w // 2, 0)
            y1 = max(center_y - roi_h // 2, 0)
            x2 = min(center_x + roi_w // 2, width)
            y2 = min(center_y + roi_h // 2, height)

            
            self.current_frame = negative_display.copy()
            cv2.rectangle(negative_display, (x1,y1), (x2,y2), (0, 255, 0), 2)

            neg_img = Image.fromarray(negative_display)
            neg_imgtk = ImageTk.PhotoImage(image=neg_img)
            self.negative_label.imgtk = neg_imgtk
            self.negative_label.config(image=neg_imgtk)

            # counting edges
            roi_frame = self.extract_roi(gray,roi_w,roi_h)
            blurred = cv2.GaussianBlur(roi_frame, (5, 5), 0)  # (kernel size, sigma)
            t1, t2 = sorted((self.canny_thresh1.get(), self.canny_thresh2.get()))
            edge = cv2.Canny(blurred, t1, t2)
            contours, _ = cv2.findContours(edge, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            edge_count = len(contours)  
            self.edge_count_label.config(text=str(edge_count))

            edge_display = cv2.cvtColor(edge, cv2.COLOR_GRAY2RGB)
            edge_img = Image.fromarray(edge_display)
            edge_imgtk = ImageTk.PhotoImage(image=edge_img)
            self.canny_label.imgtk = edge_imgtk
            self.canny_label.config(image=edge_imgtk)

            # predict
            self.predict_from_image(edge_count)

        if self.running:
            self.root.after(10, self.update_camera)

    def save_image(self):
        if self.current_frame is not None:
            save_dir = "capture"

            # Create a timestamped filename
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"frame_{timestamp}.tiff"
            file_path = os.path.join(save_dir, filename)

            # Save image
            cv2.imwrite(file_path, self.current_frame)
            # cv2.imwrite(file_path, roi)


            print(f"Image saved to {file_path}")

    def extract_roi(self, frame: np.ndarray, roi_width: int, roi_height: int) -> np.ndarray:
        height, width = frame.shape[:2]
        center_x, center_y = width // 2, height // 2
        x1 = max(center_x - roi_width // 2, 0)
        y1 = max(center_y - roi_height // 2, 0)
        x2 = min(center_x + roi_width // 2, width)
        y2 = min(center_y + roi_height // 2, height)
        return frame[y1:y2, x1:x2]
    
    def predict_from_image(self,edge_count):
        now = time.time()

        if now - self.last_pred_time > self.prediction_interval:
            # prediction
            # prediction = predict(current_frame.copy())
            # prediction_buffer.append(prediction[0])
            
            predict = 1 if edge_count < 1 else 0

            self.prediction_buffer.append(predict)
            voted_prediction = 1 if self.prediction_buffer.count(1) > self.prediction_buffer.count(-1) else -1

            # Update result label
            if voted_prediction == 1:
                self.result_text.set("✅ Unclogged")
                self.result_label.config(fg="green")

                # approaching
                if self.approaching:
                    self.send_command("m5")

                if self.deaproaching:
                    self.send_command("s")
                    self.deaproaching = False
                
                if self.guard:
                    self.releasing = False
                    pass

            else:
                self.result_text.set("⚠️ Clogged")
                self.result_label.config(fg="red")

                # deaproaching
                if self.approaching:
                    self.send_command("s")
                    self.approaching = False

                if self.deaproaching:
                    self.send_command("m-20")

                if (self.guard) & (not self.releasing):
                    self.send_command("r")
                    self.releasing = True
                
            self.last_pred_time = now

    def home(self):
        self.approaching = False
        self.deaproaching = False
        self.send_command("M0;")

    def approach(self):
        if not self.approaching:
            self.approach_text.set("✅ Approach")
            self.approach_label.config(fg="green")
            self.approaching = True
            self.deaproaching = False

        else:
            self.approach_text.set("⚠️ Stop")
            self.approach_label.config(fg="red")
            self.approaching = False
            self.deaproaching = False
            self.send_command("s")

    def guard_activate(self):
        if not self.guard:
            self.approach_text.set("✅ Guard Activate")
            self.approach_label.config(fg="green")
            self.guard = True
        else:
            self.approach_text.set("⚠️ Guard Deactivate")
            self.approach_label.config(fg="red")
            self.guard = False
            self.send_command("s")

    def deaproach(self):
        self.approaching = False
        self.deaproaching = True

    def toggle_canny(self):
        self.canny_mode = not self.canny_mode

    def update_threshold_labels(self):
        self.t1_label.config(text=str(self.canny_thresh1.get()))
        self.t2_label.config(text=str(self.canny_thresh2.get()))

    def on_close(self):
        try:
            if self.serial_port and self.serial_port.is_open:
                self.serial_port.write(b"M0;\n")
        except:
            pass
        self.running = False
        if self.cap:
            self.cap.release()
        self.running = False
        self.root.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    app = SerialMonitor(root)
    app.toggle_connection()
    root.mainloop()
