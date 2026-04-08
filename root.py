import tkinter as tk
import cv2
import matplotlib.pyplot as plt
import numpy as np
from ultralytics import YOLO
import cvzone
import my_predict_P

label1 = "Welcome to my final year project\n I'm Chan Lok Hei 22068341D\n This project aims to help enhancing the speed of classifying garbage." 

def open(): print('open')
def exit():
    print('exit')
    root.destroy()

def recreate():
    global root
    root.update()
    root.deiconify()
     
def prediction_R():
    global root
    model = YOLO("/Users/hayhay/Documents/ultralytics-main/runs/detect/train11/weights/best.pt")
    results = model.predict(source=0, show=True)
    print (results)
    if cv2.waitKey(1) == ord('z'):
        print("a")
        root.deiconify()

def prediction_P():
       global root
       root.withdraw()
       model = YOLO("/Users/hayhay/Documents/ultralytics-main/runs/detect/train11/weights/best.pt")
       img_path = '/Users/hayhay/Documents/labelling/IMG_6267.jpg'
       results = model(img_path)
       print (results)
       if cv2.waitKey(1) == ord('q'):
            root.update()
            root.deiconify()

       
# create main 
root = tk.Tk()
root.title("Project Garbage-classification")
root.geometry("700x600")

menu = tk.Menu(root)
menubar = tk.Menu(menu)
menubar.add_command(label="Open", command=root.deiconify)
menubar.add_command(label="Exit", command=exit)


# adding elements
label = tk.Label(root, text=label1, font=("Arial", 16))
label.pack(pady=150)
button1 = tk.Button(root, 
                   text="Real-Time Mode",
                   command=prediction_R)
button1.pack(side="left",padx=60)
button2 = tk.Button(root, 
                   text="Photo Analysis Mode",
                   command=prediction_P)
button2.pack(side="right",padx=60)
buttonQuit = tk.Button(root,
                       text="quit",
                       command=quit)
buttonQuit.pack(side="bottom",padx=20)

# main
root.mainloop()
    
