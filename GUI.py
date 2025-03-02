from Object_detection import OD

import cv2
import numpy as np
import tkinter as tk
import ttkbootstrap as ttk
from PIL import Image, ImageTk, ImageDraw, ImageFont



def get_x(num):
    return screen_width * (num/1366)
def get_y(num):
    return screen_height * (num/768)

def update_feed():
    ret, frame = cap.read()
    if ret:
        frame = cv2.resize(frame, (int(get_x(1270)), int(get_y(600))))
        rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb_image = OD.detect_object(rgb_image)
        pil_image = Image.fromarray(rgb_image)
        img = ImageTk.PhotoImage(image=pil_image)
        label.img = img
        label.config(image=img)
    window.after(10, update_feed)








window = ttk.Window(themename = "darkly")
window.title("Metal Vs Non-Metal")
screen_width = int(window.winfo_screenwidth())
screen_height = int(window.winfo_screenheight())
window.geometry(str(screen_width)+"x"+str(screen_height))
window.state("zoomed")
window.iconbitmap("ICON.ico")

label = ttk.Label(window)
label.place(x = get_x(50), y = get_y(50))

cap = cv2.VideoCapture(0)
update_feed()
window.mainloop()

cap.release()
cv2.destroyAllWindows()