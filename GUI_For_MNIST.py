#Importing libraries
from tkinter import *
from PIL import Image
import io
import numpy as np
import matplotlib.pyplot as plt
from keras.models import load_model
import cv2

class DigitClassifier(Frame):
    def __init__(self, parent):
        Frame.__init__(self,parent)
        self.parent = parent
        self.color = "black"
        self.brush_size = 14
        self.setUI()
    
    def setUI(self):
        self.parent.title("Digit Recognizer/Classifier")
        self.pack(fill=BOTH, expand =1)
        self.columnconfigure(6, weight =1)
        self.rowconfigure(2, weight =1)
        self.canv = Canvas(self, bg ="white")
        self.canv.grid(row=2, column=0, columnspan=7, padx=5,pady=5, sticky= E+ W+ S + N)
        self.canv.bind("<B1-Motion>",self.draw)
        clear_btn = Button(self, text="Clear All", width=10, command=lambda: self.canv.delete("all"))
        clear_btn.grid(row=0, column = 4, sticky=W)

        done_btn = Button(self, text="Done", width=10, command=lambda: self.save())
        done_btn.grid(row=0, column = 5)
    
    def save(self):
        self.canv.update()
        ps = self.canv.postscript(colormode="mono")
        img = Image.open(io.BytesIO(ps.encode('utf-8')))
        img.save("result.png")
        digit = DigitClassifier.classify()
        print(digit)
        self.show_digit(digit)
    
    @staticmethod
    def classify():
        clf = load_model("mnist_digit.h5")
        im = cv2.imread("result.png",0)
        im2 = cv2.resize(im,(28,28))
        im = im2.reshape(28,28,-1)
        im = im.reshape(1,28,28,1)
        im = cv2.bitwise_not(im)
        plt.imshow(im.reshape(28,28),cmap="Greys")
        result = clf.predict(im)
        digit = np.argmax(result)
        return digit
    
    def show_digit(self, digit):
        text_label= Label(self, text =digit)
        text_label.grid(row=0,column=6, padx=5, pady=5)

    def draw(self, event):
        self.canv.create_oval(event.x-self.brush_size,
            event.y-self.brush_size,
            event.x+self.brush_size,
            event.y+self.brush_size,
            fill=self.color,outline=self.color)


def mainFunc():
    root = Tk()
    root.geometry("400x400")
    root.resizable(0, 0)
    app = DigitClassifier(root)
    root.mainloop()

if __name__ == '__main__':
    mainFunc()