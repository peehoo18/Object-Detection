import csv
import os
import threading
import time
import PIL.Image
import PIL.ImageTk
import cv2
from tkinter import Tk, Frame, Canvas, Button, PanedWindow, Label, Scrollbar, messagebox
from tkinter.messagebox import showinfo, showerror
import torch

model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
destination = 'details.txt'


class CentroidTracker:
    def __init__(self):
        self.next_object_id = 0
        self.objects = {}

    def update(self, detections):
        updated_objects = {}

        for box in detections:
            (x1, y1, x2, y2) = box
            centroid_x = (x1 + x2) // 2
            centroid_y = (y1 + y2) // 2
            centroid = (centroid_x, centroid_y)

            # Check if the object has been seen before
            object_id = self.get_object_id(centroid, box)

            # Update the object dictionary
            updated_objects[object_id] = (centroid, box)

        self.objects = updated_objects
        return self.objects

    def get_object_id(self, centroid, box):
        # Check if the centroid is close to any existing object centroids and has overlapping bounding boxes
        for object_id, object_data in self.objects.items():
            object_centroid, object_box = object_data
            distance = self.calculate_distance(centroid, object_centroid)
            iou = self.calculate_iou(box, object_box)

            if distance < 50 and iou > 0.5:  # Set threshold values for matching centroids and bounding boxes
                return object_id

        # If no match found, assign a new object ID
        self.next_object_id += 1
        return self.next_object_id

    def calculate_distance(self, centroid1, centroid2):
        x1, y1 = centroid1
        x2, y2 = centroid2
        distance = ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5
        return distance

    def calculate_iou(self, box1, box2):
        x1_1, y1_1, x2_1, y2_1 = box1
        x1_2, y1_2, x2_2, y2_2 = box2

        intersect_x1 = max(x1_1, x1_2)
        intersect_y1 = max(y1_1, y1_2)
        intersect_x2 = min(x2_1, x2_2)
        intersect_y2 = min(y2_1, y2_2)

        intersect_width = max(0, intersect_x2 - intersect_x1 + 1)
        intersect_height = max(0, intersect_y2 - intersect_y1 + 1)

        intersect_area = intersect_width * intersect_height

        box1_area = (x2_1 - x1_1 + 1) * (y2_1 - y1_1 + 1)
        box2_area = (x2_2 - x1_2 + 1) * (y2_2 - y1_2 + 1)

        iou = intersect_area / float(box1_area + box2_area - intersect_area)
        return iou

class VideoFrame:

    def __init__(self, video_source=0, width=None, height=None, fps=None):

        self.video_source = video_source
        self.width = width
        self.height = height
        self.fps = fps

        # Open the video source
        self.vid = cv2.VideoCapture(video_source)

        if not self.vid.isOpened():
            raise ValueError("[VideoFrame] Unable to open video source", video_source)

        # Get video source width and height
        if not self.width:
            self.width = int(self.vid.get(cv2.CAP_PROP_FRAME_WIDTH))  # convert float to int
        if not self.height:
            self.height = int(self.vid.get(cv2.CAP_PROP_FRAME_HEIGHT))  # convert float to int
        if not self.fps:
            self.fps = int(self.vid.get(cv2.CAP_PROP_FPS))  # convert float to int

        # default value at start
        self.ret = False
        self.frame = None

        # start thread
        self.running = True
        self.thread = threading.Thread(target=self.process)
        self.thread.start()

    def process(self):
        while self.running:
            ret, frame = self.vid.read()

            if ret:
                # process image
                frame = cv2.resize(frame, (self.width, self.height))
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            else:
                print('[VideoFrame] stream end:', self.video_source)
                # TODO: reopen stream
                self.running = False
                break

            # assign new frame
            self.ret = ret
            self.frame = frame

            # sleep for next frame
            time.sleep(1 / self.fps)

    def get_frame(self):
        return self.ret, self.frame

    # Release the video source when the object is destroyed
    def __del__(self):
        # stop thread
        if self.running:
            self.running = False
            self.thread.join()

        # release stream
        if self.vid.isOpened():
            self.vid.release()


class Camera(Frame):

    def __init__(self, window, canvas, video_source=0, width=None, height=None):
        super().__init__(canvas)
        self.photo = None
        self.window = window
        self.canvas = canvas
        self.video_source = video_source
        self.vid = VideoFrame(self.video_source, width, height)
        self.delay = int(1000 / self.vid.fps)

        print('[Camera] source:', self.video_source)
        print('[Camera] fps:', self.vid.fps, 'delay:', self.delay)
        self.canvas.bind('<Button-1>', self.on_mouse)
        self.image = None
        self.rect = (0, 0, 0, 0)
        self.str_pt = False
        self.end_pt = False
        self.roi_list = []
        self.roi_trackers = {}
        self.running = True

        self.update_frame()

    def on_mouse(self, event ):
        # global rect, str_pt, end_pt, roi_list
        x = self.canvas.canvasx(event.x)
        y = self.canvas.canvasy(event.y)
        if event.type == '4':
            if self.str_pt is True and self.end_pt is True:
                self.str_pt = False
                self.end_pt = False
                self.rect = (0, 0, 0, 0)
            if self.str_pt is False:
                self.rect = (x, y, 0, 0)
                self.str_pt = True
            elif self.end_pt is False:
                self.rect = (self.rect[0], self.rect[1], x, y)
                self.roi_list.append(self.rect)
                self.roi_trackers[len(self.roi_list) - 1] = CentroidTracker()  # Create a tracker for the new ROI
                self.end_pt = True

    def update_frame(self):
        # Get a frame from the video source
        ret, frame = self.vid.get_frame()
        detections = []
        d = None  # Initialize d variable
        conf = None
        if self.str_pt == True and self.end_pt == True:
            for roi_index, roi in enumerate(self.roi_list):
                x, y, w, h = roi
                x = int(x)
                y = int(y)
                w = int(w)
                h = int(h)
                cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 255), 2)
                roi = frame[y:h, x:w]
                results = model(roi)

                for index, row in results.pandas().xyxy[0].iterrows():
                    x1 = int(row['xmin'])
                    y1 = int(row['ymin'])
                    x2 = int(row['xmax'])
                    y2 = int(row['ymax'])
                    conf = row['confidence']*100
                    d = (row['name'])
                    print("Name: ", d, "\t confidence: ", conf)
                    if x1 >= x and y1 >= y and x2 <= w and y2 <= h and x1 >= 0 and y1 >= 0 and x2 <= frame.shape[1] and \
                            y2 <= frame.shape[0]:
                        detections.append([x1, y1, x2, y2])
                        cv2.rectangle(roi, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(roi, d, (x1 + 15, y1 + 15), cv2.FONT_HERSHEY_COMPLEX, 0.75, (0, 255, 0), 1)
                objects = self.roi_trackers[roi_index].update(detections)
                for objectID, centroid in objects.items():
                    note = f"ID: {objectID}, Name: {d}, Confidence: {conf:.2f}"
                    if os.path.isfile(destination):
                        try:
                            with open(destination, "at", encoding="utf-8") as file:
                                file.write("\n" + note)
                                print("Success", "Previous entry edited")
                        except:
                            print("An exception occured while editing a previous file.")
                            showerror("Error", "An exception occured while editing a previous file.")
                    else:
                        try:
                            with open(destination, "wt", encoding="utf-8") as file:
                                file.write(note)
                                print("Success", "New entry saved")
                        except:
                            print("An exception occured while writing a new file.")
                            showerror("Error", "An exception occured while writing a new file.")

        if ret:
            self.image = PIL.Image.fromarray(frame)
            self.photo = PIL.ImageTk.PhotoImage(image=self.image)
            self.canvas.create_image(0, 0, image=self.photo, anchor='nw')

        if self.running:

            self.canvas.after(self.delay, self.update_frame)


class Window:

    def __init__(self, window, window_title, video_source):
        self.frames = []
        self.selected_link = None
        self.window = window
        self.video_sources = video_source
        screen_width = window.winfo_screenwidth()
        screen_height = window.winfo_screenheight()
        window.geometry(f'{screen_width}x{screen_height}+0+0')
        print(screen_height, screen_width)
        window.configure(background="grey2")
        window.title(window_title)
        # cv2.namedWindow("LIVE CAM", cv2.WINDOW_NORMAL)
        # cv2.setMouseCallback("LIVE CAM", on_mouse)
        canvas_list = []

        panel_1 = PanedWindow(bd=4, relief="raised", bg="red")
        panel_1.pack(fill="both", expand=1)

        left_label = Frame(panel_1, width=210)
        panel_1.add(left_label, stretch='never')
        label_1 = Label(left_label, width=20, height=3, text='Choose the Frame')
        label_1.pack()

        v_scroll = Scrollbar(left_label, orient='vertical')

        canvas_1 = Canvas(left_label, yscrollcommand=v_scroll.set)
        canvas_1.pack(side='left', fill='both', expand=True)

        # Configure the canvas_1 to use the scrollbar for scrolling
        v_scroll.config(command=canvas_1.yview)
        v_scroll.pack(side='right', fill='y')
        canvas_1.config(scrollregion=canvas_1.bbox("all"))

        panel_2 = PanedWindow(panel_1, width=1057, orient="vertical", bd=4, relief="raised", bg="purple")
        panel_1.add(panel_2, stretch='never')

        top = Frame(panel_2, height=1000, width=1057, background="grey")
        panel_2.add(top, stretch='never')

        bottom1 = Frame(panel_2, height=3)
        panel_2.add(bottom1, stretch='never')

        one = Button(bottom1, text=f'1X1', command=lambda: self.grid(top, left_label, 1, canvas_list=[]))
        one.grid(row=0, column=0)
        two = Button(bottom1, text=f'2X2', command=lambda: self.grid(top, left_label, 2, canvas_list=[]))
        two.grid(row=0, column=1)
        three = Button(bottom1, text=f'3X3', command=lambda: self.grid(top, left_label, 3, canvas_list=[]))
        three.grid(row=0, column=2)
        four = Button(bottom1, text=f'4X4', command=lambda: self.grid(top, left_label, 4, canvas_list=[]))
        four.grid(row=0, column=3)
        five = Button(bottom1, text=f'5X5', command=lambda: self.grid(top, left_label, 5, canvas_list=[]))
        five.grid(row=0, column=4)
        six = Button(bottom1, text=f'6X6', command=lambda: self.grid(top, left_label, 6, canvas_list=[]))
        six.grid(row=0, column=5)

        # bottom2 = Label(panel_2, height=10, text="Bottom1 Panel")
        # panel_2.add(bottom2, stretch='never')

        right_label = Frame(panel_1, width=200)
        panel_1.add(right_label, stretch='never')
        button_add = Button(right_label, width=20, height=3, text='Add New Camera',
                            command=lambda: self.add_new(right_label))
        button_add.pack()
        label_2 = Label(right_label, width=20, height=3, text='Choose the Camera')
        label_2.pack()
        for names in self.video_sources:
            if len(names) >= 2:
                text, link = names
                button = Button(right_label, width=20, text=f'{text}', command=lambda l=link: self.select_link(l))
                button.pack()
            else:
                continue

        ''''''
        self.window.protocol("WM_DELETE_WINDOW", self.on_closing)
        self.window.state('zoomed')
        self.window.mainloop()

    def add_new(self, right_label):
        camera_name = input("Enter the Camera Number: ")
        camera_link = input("Enter the Camera Link: ")
        video = (camera_name, camera_link)
        with open('camera_details.csv', 'a', newline='') as file_1:
            writer_object = csv.writer(file_1)
            writer_object.writerow(video)
        with open('camera_details.csv', 'r', newline='') as file_1:
            source = [tuple(line) for line in csv.reader(file_1)]
            self.video_sources = source
            for widget in right_label.winfo_children():
                widget.destroy()
            label_2 = Label(right_label, width=20, height=3, text='Choose the Camera')
            label_2.pack()
            for names in self.video_sources:
                if len(names) >= 2:
                    text, link = names
                    button = Button(right_label, width=20, text=f'{text}', command=lambda l=link: self.select_link(l))
                    button.pack()
                else:
                    continue

    def grid(self, top, left_label, n, canvas_list=None, button_list=None):
        canvas_width = 1057 // n
        canvas_height = 1000 // n
        # Clear previously created buttons and labels
        for widget in left_label.winfo_children():
            widget.destroy()

        label_1 = Label(left_label, width=30, height=3, text='Choose the Frame')
        label_1.pack()

        v_scroll = Scrollbar(left_label, orient='vertical')
        v_scroll.pack(side='right', fill='y')

        canvas_1 = Canvas(left_label, width=30, yscrollcommand=v_scroll.set)
        canvas_1.pack()

        v_scroll.config(command=canvas_1.yview)

        # canvas_1.config(scrollregion=canvas_1.bbox('all'))

        for widget in top.winfo_children():
            widget.destroy()

        for row in range(n):
            top.rowconfigure(row, weight=1)
            for col in range(n):
                top.columnconfigure(col, weight=1)
                canvas = Canvas(top, height=canvas_height, width=canvas_width, background="black")
                canvas.grid(row=row, column=col, sticky='nsew')
                canvas_list.append(canvas)
        for j in range(1, (n * n) + 1):
            button1 = Button(canvas_1, width=20, text=f'{j}', command=lambda num=j: prints(num))
            button1.pack()

        def prints(i):
            print(i)
            print(self.selected_link)
            if self.selected_link is None:
                print("Error: Camera not selected")
                print("First select the camera then chose the frame ")
                showerror('Error', 'Error: Camera not selected \n First select the camera then chose the frame')
            else:
                if self.selected_link in self.frames:
                    showinfo('Message', "Already selected")
                    print("Already selected")
                else:
                    self.frames.append(self.selected_link)
                    selected_frame = canvas_list[i - 1]
                    print(selected_frame)
                    Camera(top, selected_frame, self.selected_link, canvas_width, canvas_height)

    def select_link(self, link):
        self.selected_link = link

    def on_closing(self):
        print('[Window] exit')
        self.window.destroy()


if __name__ == '__main__':
    # sources = []
    with open('camera_details.csv', 'r', newline='') as file:
        sources = [tuple(line) for line in csv.reader(file)]
        '''
    with open('camera_details.csv', 'r') as file:
        read = csv.reader(file)
        for row in read:
            if len(row) >= 2:  # Check if the row has at least 2 columns
                filename = row[0]
                other_data = row[1]
                sources.append((filename, other_data))
            else:
                print(f"Skipping row: {row}")'''
        '''
    sources = [
        # ('Camera 1', 'http://webcam01.ecn.purdue.edu/mjpg/video.mjpg'),
        # ('Camera 2', 'http://pendelcam.kip.uni-heidelberg.de/mjpg/video.mjpg'),
        # ('Camera 3', 'http://61.211.241.239/nphMotionJpeg?Resolution=320x240&Quality=Standard'),
        # ('Camera 4', 'http://pendelcam.kip.uni-heidelberg.de/mjpg/video.mjpg'),
        ('Camera 5', '1.mp4'), ('Camera 6', 'highway.mp4'),
        ('Camera 7', 'pexels-mike-bird-2053100-3840x2160-60fps.mp4'),
        ('Camera 8', 'pexels-mike-bird-2103099-3840x2160-60fps.mp4'),
        ('Camera 9', 'pexels-mostafa-meraji-3078508-1920x1080-30fps.mp4'),
        ('Camera 10', 'pexels-nino-souza-2099536-1920x1080-60fps.mp4'),
        ('Camera 11', 'pexels-nino-souza-2252223-3840x2160-30fps.mp4'),
        ('Camera 12', 'pexels-taryn-elliott-3121459-3840x2160-24fps.mp4')
    ] '''
    Window(Tk(), "LIVE CAM", sources)
