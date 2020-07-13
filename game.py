import sys
import os
import tkinter as tk
from PIL import Image, ImageTk


#User interface of application, written by Aliaksandr Nenartovich
def play():
	os.system("python3 train.py")

def make_video():
	os.system("python3 evaluate.py")

root = tk.Tk()
root.geometry("700x550")
root.title("Multi-Agent Interaction")

def quit():
	root.destroy()

btn1 = tk.Button(root, text="Run Program", command=play)
video = tk.Button(root, text="Create Video", command=make_video)
quit = tk.Button(root, text="Quit", command=quit)

background_image=ImageTk.PhotoImage(Image.open("images/landing.gif").resize((683, 384)))
background_label = tk.Label(root, image=background_image)

#Add text
text = tk.Label(root, text="Welcome to Multi-Agent Interaction!")

#Change text size, from https://stackoverflow.com/questions/30685308/how-do-i-change-the-text-size-in-a-label-widget-python-tkinter
text.config(font=("Arial", 25))

text.pack(pady = (0,0))
background_label.pack()
btn1.pack(pady=(0,10))
video.pack(pady=(0,10))
quit.pack(pady=(0,20))

root.mainloop()