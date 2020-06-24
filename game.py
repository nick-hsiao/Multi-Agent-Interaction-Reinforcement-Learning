import sys
import os
import tkinter as tk
from PIL import Image, ImageTk

def play():
	os.system("python3 train.py")

def make_video():
	os.system("python3 evaluate.py")

root = tk.Tk()
root.geometry("550x530")
root.title("GUI Button")

def quit():
	root.destroy()

btn1 = tk.Button(root, text="Run Program", command=play)
video = tk.Button(root, text="Create Video", command=make_video)
quit = tk.Button(root, text="Quit", command=quit)

background_image=ImageTk.PhotoImage(Image.open("images/landing.gif").resize((550, 400)))
background_label = tk.Label(root, image=background_image)

background_label.pack()
btn1.pack(pady=(0,10))
video.pack(pady=(0,10))
quit.pack(pady=(0,20))

root.mainloop()