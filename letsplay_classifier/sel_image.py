
import tkinter as tk
from PIL import Image, ImageTk
import json
import os
from util import retrieve_or_create_dict
import argparse


THRESHOLD = 2.5

class_names = ['Battle', 'Hideout', 'Other', 'Siege', 'Tournament']
data_dir = '../wendy_cnn_frames_data_2'

rejected = retrieve_or_create_dict('rejected.json')
misclassified = retrieve_or_create_dict('misclassified.json')
confirmed = retrieve_or_create_dict('confirmed.json')

class App:
    """
    Helper small application to check on images that may have been misclassified by a model. Will show an        image and ask the user to select between predicted and expected label.
    """

    def __init__(self, master=tk.Tk(), data_dir=data_dir, class_names=class_names, threshold=THRESHOLD):

        self.master = master
        self.data_dir = data_dir
        self.class_names = class_names
        self.threshold = threshold
        self.all_images = list(get_next_image(self.threshold))
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

        while True:
            if self.images_index >= len(self.all_images):
                self.close()

            label_pred, (file_name, prediction_prob) = self.all_images[self.images_index]
            self.images_index += 1

            if not (file_name in confirmed or file_name in rejected):
                break
        label_index, pred_index = map(int, (label_pred.split(':')))

        file_name_full = os.path.join(self.data_dir, self.class_names[label_index], file_name)
        self.button_left.config(text=self.class_names[label_index])
        self.button_right.config(text=self.class_names[pred_index])
        self.load_image(file_name_full)
        self.image_label.config(image=self.fig_image)
        self.maxx_label.config(text=prediction_prob)
        return file_name, label_index, pred_index

    def close(self, *args):
        self.master.quit()
        self.is_active = False

    def is_closed(self):
        return not self.is_active

    def mainloop(self):
        self.master.mainloop()


def get_next_image(threshold=THRESHOLD):
    """
    generator going through misclassified images
    :return: yields in a generator all misclassified images
    """
    for key in misclassified.keys():
        values = misclassified[key]
        for value in values:
            if (value[1] > threshold):
                yield key, value


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--data-dir', type=str, default=os.environ['SM_CHANNEL_TRAIN'])
    parser.add_argument('--threshold', type=float, default=2.5, metavar='N',
                        help='min log probability of true value for checking on image')

    args = parser.parse_args()
    class_names = [s for s in sorted(os.listdir(args.data_dir)) if os.path.isdir(os.path.join(args.data_dir, s))]
    app = App(data_dir=args.data_dir, class_names=class_names, threshold=args.threshold)
    app.mainloop()


