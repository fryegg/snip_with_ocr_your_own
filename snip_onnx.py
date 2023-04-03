import pyautogui
from PIL import Image, ImageDraw, ImageTk
from tkinter import *
import numpy as np
from matplotlib import pyplot as plt
from PIL import ImageFont, ImageDraw, Image
import logging
from tkinter import messagebox
import easyocr_only_onnx.easyocr as easyocr

class SnippingTool:
    def __init__(self, output_format='text', output_file=None):
        super().__init__()
        self.root = Tk()
        self.root.withdraw()
        self.root.after(0, self.start_snipping)
        self.root.mainloop()

        self.output_format = output_format
        self.output_file = output_file

    def start_snipping(self):
        self.root.deiconify()
        self.root.attributes('-alpha', 0.3)
        self.root.attributes('-fullscreen', True)
        self.canvas = Canvas(self.root, bg='white')
        self.canvas.pack(fill=BOTH, expand=True)
        self.canvas.bind('<ButtonPress-1>', self.on_button_press)
        self.canvas.bind('<B1-Motion>', self.on_move_press)
        self.canvas.bind('<ButtonRelease-1>', self.on_button_release)
        self.canvas.bind('<ButtonRelease-3>', self.on_exit_press)
        #self.canvas.bind("<ButtonRelease-1>", self.show_popup)
        self.x = self.y = 0
        self.rect = None

    def on_button_press(self, event):
        self.x, self.y = event.x, event.y
        self.rect = None

    def on_move_press(self, event):
        if self.rect:
            self.canvas.delete(self.rect)
        self.rect = self.canvas.create_rectangle(self.x, self.y, event.x, event.y, outline='black', width=2)

    # def putText(self,cv_img, text, x, y, color=(0, 0, 0), font_size=22):
    #     # Colab이 아닌 Local에서 수행 시에는 gulim.ttc 를 사용하면 됩니다.
    #     # font = ImageFont.truetype("fonts/gulim.ttc", font_size)
    #     #font = ImageFont.truetype('나눔_글꼴/나눔고딕/NanumFontSetup_TTF_GOTHIC/NanumGothicExtraBold.ttf', font_size)
    #     img = Image.fromarray(cv_img)
        
    #     draw = ImageDraw.Draw(img)
    #     draw.text((x, y), text, font=font, fill=color)
        
    #     cv_img = np.array(img)
        
    #     return cv_img

    def on_button_release(self, event):
        self.root.attributes('-alpha', 0.0)
        self.root.withdraw()
        try:
            x1, y1 = self.root.winfo_rootx() + self.x, self.root.winfo_rooty() + self.y
            x2, y2 = self.root.winfo_rootx() + event.x, self.root.winfo_rooty() + event.y
            im = pyautogui.screenshot(region=(x1, y1, x2 - x1, y2 - y1) if x2 > x1 else (x2, y2, x1 - x2, y1 - y2))
            im = np.array(im)
            #reader = easyocr.Reader(['ko','en']) # this needs to run only once to load the model into memory
            #reader = easyocr.Reader(['en'], detector='DB', recognizer='Transformer')
            reader = easyocr.Reader(['en'], gpu=True)
            results = reader.readtext(im)
            # results = reader.readtext(im)
            #popup
            self.popup = Toplevel(self.canvas)
            self.popup.geometry("400x400")
            self.popup.title("Popup Window")

            # Create a listbox
            listbox = Listbox(self.popup,width=100, height=20)
            #listbox.pack(side="left", fill="y")

            scrollbar = Scrollbar(self.popup, orient="vertical")
            scrollbar.config(command=listbox.yview)
            scrollbar.pack(side="right", fill="y")

            listbox.config(yscrollcommand=scrollbar.set)
            # loop over the results
            for (bbox, text, prob) in results:
                print("[INFO] {:.4f}: {}".format(prob, text))
                
                (tl, tr, br, bl) = bbox
                tl = (int(tl[0]), int(tl[1]))
                tr = (int(tr[0]), int(tr[1]))
                br = (int(br[0]), int(br[1]))
                bl = (int(bl[0]), int(bl[1]))
                listbox.insert(END, text)
                #im = self.putText(im, text, tl[0], tl[1] + 20, (0, 255, 0), 10)
            #self.plt_imshow("Image", im, figsize=(16,10))
            listbox.pack()
            # Bind the listbox to hide the popup on selection
            self.popup.protocol("WM_DELETE_WINDOW", self.hide_popup)
        except Exception as e:
            logging.exception("An exception was thrown!")
            self.root.quit()
    def hide_popup(self):
        print("hello?")
        if messagebox.askokcancel("Quit", "Do you want to quit?"):
            self.root.destroy()
            self.root.quit()
    def on_exit_press(self, event):
        self.root.quit()

if __name__ == '__main__':
    snipping_tool = SnippingTool(output_format='draw', output_file='output.txt')