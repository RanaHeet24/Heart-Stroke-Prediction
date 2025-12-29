import tkinter as tk
from tkinter import messagebox

def calculate(op):
    try:
        a = float(entry1.get())
        b = float(entry2.get())

        if op == '+':
            result = a + b
        elif op == '-':
            result = a - b
        elif op == '*':
            result = a * b
        elif op == '/':
            if b == 0:
                messagebox.showerror("Error", "Division by zero not allowed!")
                return
            result = a / b

        label_result.config(text=f"Result: {result}")
    except ValueError:
        messagebox.showerror("Invalid Input", "Please enter valid numbers!")

# Main Window
root = tk.Tk()
root.title("Mini Calculator - Tkinter")
root.geometry("350x350")
root.config(bg="#f0f4f7")

# Title Label
title = tk.Label(root, text="🧮 Simple Calculator", font=("Arial", 16, "bold"), bg="#f0f4f7", fg="#333")
title.pack(pady=15)

# Input Fields
tk.Label(root, text="Enter First Number:", font=("Arial", 12), bg="#f0f4f7").pack(pady=5)
entry1 = tk.Entry(root, width=25, font=("Arial", 12))
entry1.pack()

tk.Label(root, text="Enter Second Number:", font=("Arial", 12), bg="#f0f4f7").pack(pady=5)
entry2 = tk.Entry(root, width=25, font=("Arial", 12))
entry2.pack()

# Buttons Frame
frame = tk.Frame(root, bg="#f0f4f7")
frame.pack(pady=20)

btn_style = {"font": ("Arial", 12, "bold"), "width": 6, "height": 1, "bg": "#4CAF50", "fg": "white"}

tk.Button(frame, text="+", command=lambda: calculate('+'), **btn_style).grid(row=0, column=0, padx=5, pady=5)
tk.Button(frame, text="-", command=lambda: calculate('-'), **btn_style).grid(row=0, column=1, padx=5, pady=5)
tk.Button(frame, text="×", command=lambda: calculate('*'), **btn_style).grid(row=1, column=0, padx=5, pady=5)
tk.Button(frame, text="÷", command=lambda: calculate('/'), **btn_style).grid(row=1, column=1, padx=5, pady=5)

# Result Label
label_result = tk.Label(root, text="Result: ", font=("Arial", 14, "bold"), bg="#f0f4f7", fg="#222")
label_result.pack(pady=20)

# Exit Button
tk.Button(root, text="Exit", font=("Arial", 12, "bold"), bg="#f44336", fg="white", width=10, command=root.destroy).pack(pady=10)

root.mainloop()
