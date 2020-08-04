import sys
import os
import tkinter as tk
from PIL import Image, ImageTk
import subprocess

#User interface of application, written by Aliaksandr Nenartovich
def play():
	subprocess.call(["python3", "train.py", size_entered.get(), walls_entered.get(), agents_entered.get(), agents_entered.get()])

def make_video():
	subprocess.call(["python3", "evaluate.py", size_entered.get(), walls_entered.get(), agents_entered.get(), agents_entered.get()])

root = tk.Tk()
root.geometry("800x610")
root.title("Multi-Agent Interaction")

def quit():
	root.destroy()

grid_label = tk.Label(root, text = "Grid Size:")
grid = tk.StringVar()
size_entered = tk.Entry(root, width=4, textvariable = grid)
wall_label = tk.Label(root, text = "Number of Walls:")
walls = tk.StringVar()
walls_entered = tk.Entry(root, width = 4, textvariable = walls)
agent_label = tk.Label(root, text = "Number of Agents Per Team:")
agents = tk.StringVar()
agents_entered = tk.Entry(root, width=4, textvariable = agents)
# def_agent_label = tk.Label(root, text = "Number of Defender Agents:")
# def_agents = tk.StringVar()
# def_agents_entered = tk.Entry(root, width=4, textvariable = def_agents)


btn1 = tk.Button(root, text="Run Program", command=play)
video = tk.Button(root, text="Create Video", command=make_video)
quit = tk.Button(root, text="Quit", command=quit)

background_image=ImageTk.PhotoImage(Image.open("images/landing.gif").resize((683, 384)))
background_label = tk.Label(root, image=background_image)

#Add text
text = tk.Label(root, text="Welcome to Multi-Agent Interaction!")
enter = tk.Label(root, text="Please Fill in All Fields Below:")

#Change text size, from https://stackoverflow.com/questions/30685308/how-do-i-change-the-text-size-in-a-label-widget-python-tkinter
text.config(font=("Arial", 25))
enter.config(font=("Arial", 20))
background_label.configure(background='black')

text.pack(pady = (0,0))
background_label.pack()

enter.pack(pady=(30,0))
grid_label.pack(padx = (50,0), side = tk.LEFT)
size_entered.pack(padx = (5,40), side = tk.LEFT)
wall_label.pack(padx = 0, side = tk.LEFT)
walls_entered.pack(padx = (5,40), side = tk.LEFT)
agent_label.pack(padx = 0, side = tk.LEFT)
agents_entered.pack(padx = 0, side = tk.LEFT)
# def_agent_label.pack(padx = 0, side = tk.LEFT)
# def_agents_entered.pack(padx = 0, side = tk.LEFT)

btn1.pack(pady=(0,10))
video.pack(pady=(0,10))
quit.pack(pady=(0,10))

root.mainloop()
