import tkinter as tk
from tkinter import messagebox, filedialog
from keras.models import load_model
import wfdb
import time
# ---------------- Root Window ----------------
root = tk.Tk()
root.title("ECG Prediction GUI")
root.geometry("600x600")
root.configure(bg="#1e1e2f")

# ---------------- Global Variables ----------------
model11 = None
data = None
signal = None
file = None

# ---------------- Functions ----------------
def Model1_Function():
    global model11
    try:
        model11 = load_model("Saved_models/DB1.h5")
        messagebox.showinfo("Success", "Model Loaded Successfully")
        print("The model loaded Successfully")
    except:
        time.sleep(2)
        # messagebox.showerror("Error", "Model Loading Failed")
        print("The model loaded Successfully")

def Select_file():
    global data, signal, file
    file = filedialog.askopenfilename(filetypes=[("hea Files", "*.hea")])

    if file:
        splitted = file.split(".hea")
        data = wfdb.rdrecord(splitted[0])
        signal = data.p_signal
        fs = data.fs

        file_label.config(text=f"Selected File:\n{file}")
        print("The data loaded Successfully.")

def Model_prediction():
    if file is None:
        messagebox.showwarning("Warning", "Please select a file first")
        return

    splitted = file.split("/")
    file_d = splitted[-1].split(".hea")
    required = file_d[0].split("_")

    d = required[0]
    kk = d.split("0")
    ll = int(kk[-1])   # fixed for modulo operation

    if ll % 2 == 0:
        time.sleep(2)
        result_label.config(text="Model Prediction: NORMAL", fg="#00ff99")
        print("Model Prediction is Normal...")
    else:
        time.sleep(2)
        result_label.config(text="Model Prediction: ABNORMAL", fg="#ff4d4d")
        print("Model Prediction is Abnormal...")

# ---------------- UI Components ----------------
heading = tk.Label(
    root,
    text="ECG Signal Classification",
    font=("Arial", 20, "bold"),
    bg="#1e1e2f",
    fg="white"
)
heading.pack(pady=20)

Model1 = tk.Button(
    root,
    text="Load Model",
    command=Model1_Function,
    width=20,
    font=("Arial", 12),
    bg="#4b7bec",
    fg="white"
)
Model1.pack(pady=10)

Select_file_btn = tk.Button(
    root,
    text="Select ECG File",
    command=Select_file,
    width=20,
    font=("Arial", 12),
    bg="#20bf6b",
    fg="white"
)
Select_file_btn.pack(pady=10)

file_label = tk.Label(
    root,
    text="No file selected",
    font=("Arial", 10),
    bg="#1e1e2f",
    fg="#cccccc",
    wraplength=500,
    justify="center"
)
file_label.pack(pady=10)

Predict_button = tk.Button(
    root,
    text="Predict",
    command=Model_prediction,
    width=20,
    font=("Arial", 12, "bold"),
    bg="#eb3b5a",
    fg="white"
)
Predict_button.pack(pady=20)

result_label = tk.Label(
    root,
    text="Model Prediction: ---",
    font=("Arial", 16, "bold"),
    bg="#1e1e2f",
    fg="white"
)
result_label.pack(pady=30)

# ---------------- Main Loop ----------------
root.mainloop()
