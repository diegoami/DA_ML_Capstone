
import tkinter as tk
from PIL import Image, ImageTk
import json
import os
import itertools
import multiprocessing
class_names = ['Battle', 'Hideout', 'Other', 'Siege', 'Tournament']
data_dir = '../wendy_cnn_frames_data'
if os.path.isfile('misclassified.json'):
    with open('misclassified.json', 'r') as f:
        misclassified = json.load(f)

if os.path.isfile('confirmed.json'):
    with open('confirmed.json', 'r') as f:
        confirmed = json.load(f)
else:
    confirmed = {}

if os.path.isfile('rejected.json'):
    with open('rejected.json', 'r') as f:
        rejected = json.load(f)
else:
    rejected = {}


class App:
    def __init__(self, master=tk.Tk(), image_name=''):

        self.master = master
        self.all_images = list(get_next_image())
        self.images_index = 0
        self.fig_size = [1200, 720]
        self.frame = tk.Frame(master)
        self.canvas = tk.Canvas(self.frame, width=1280, height=800)
        self.canvas.pack()

        self.load_image()

        self.image_label = tk.Label(self.canvas, image=self.fig_image)
        self.image_label.pack()
        self.maxx_label = tk.Label(self.canvas)
        self.maxx_label.pack(side="bottom")

        self.button_left = tk.Button(self.frame, text="BUTTON",
                                     command=self.update_left)
        self.button_left.pack(side="left")

        self.button_right = tk.Button(self.frame, text="BUTTON",
                                     command=self.update_right)
        self.button_right.pack(side="right")
        self.frame.bind("q", self.close)
        self.frame.bind("<Escape>", self.close)
        self.frame.pack()
        self.frame.focus_set()

        self.is_active = True



    def load_image(self, filename=None):
        if (filename):
            self.fig_image = ImageTk.PhotoImage(Image.open(filename).resize(self.fig_size, Image.BILINEAR))
        else:
            self.fig_image = ImageTk.PhotoImage("RGB", size=self.fig_size)

    def update_left(self, *args):
        file_name, left_index, right_index = self.update()
        confirmed[file_name] = left_index
        with open('confirmed.json', 'w') as f:
            json.dump(confirmed, f)

    def update_right(self, *args):
        file_name, left_index, right_index = self.update()
        rejected[file_name] = right_index
        with open('rejected.json', 'w') as f:
            json.dump(rejected, f)

    def update(self, *args):
        file_name = ''
        while True:
            self.images_index += 1

            key, (file_name, maxx) = self.all_images[self.images_index]
            if not (file_name in confirmed or file_name in rejected):
                break
        label_index, pred_index = map(int,(key.split(':')))

        file_name_full = os.path.join(data_dir, class_names[label_index], file_name)
        self.button_left.config(text=class_names[label_index])
        self.button_right.config(text=class_names[pred_index])
        self.load_image(file_name_full)
        self.image_label.config(image=self.fig_image)
        self.maxx_label.config(text=maxx)
        return file_name, label_index, pred_index

    def close(self, *args):
        print('GUI closed...')
        self.master.quit()
        self.is_active = False

    def is_closed(self):
        return not self.is_active



    def mainloop(self):
        self.master.mainloop()


        print('mainloop closed...')

def get_next_image():
    for key in misclassified.keys():
        values = misclassified[key]
        for value in values:
            if (value[1] > 4):
            #print(key, value)
                yield key, value
if __name__ == '__main__':
    import time



    app = App()
    app.mainloop()

    #print(next(get_next_image()))
    #print(next(get_next_image()))
    #print(next(get_next_image()))
