import cv2
from typing import Any
from dataclasses import dataclass

@dataclass    
class VideoReader:
    source : str 
    reader : Any = None
        
    def initialize(self):
        self.reader = cv2.VideoCapture(self.source)
    
    def next_frame(self):
        frame = None
        if self.reader.isOpened():
            _, frame = self.reader.read()
        return frame
    
    def is_running(self):
        return self.reader.isOpened()
    
    def close(self):
        self.reader.release()
        cv2.destroyAllWindows()
        
    def get_video_width(self):
        width = -1 
        if self.reader.isOpened():
            width = int(self.reader.get(cv2.CAP_PROP_FRAME_WIDTH))
        return width
    
    def get_video_height(self):
        height = -1
        if self.reader.isOpened():
            height = int(self.reader.get(cv2.CAP_PROP_FRAME_HEIGHT))
        return height
    
    def set_window_size(self, dims):
        self.reader.set(cv2.CAP_PROP_FRAME_WIDTH, dims[0])
        self.reader.set(cv2.CAP_PROP_FRAME_HEIGHT, dims[1])
        

@dataclass
class VideoWriter:
    dest : str
    writer : Any = None 
    
    def initialize(self, width, height):
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        self.writer = cv2.VideoWriter(self.dest, fourcc, 20.0, (width, height))
        
    def write(self, frame):
        if self.writer:
            self.writer.write(frame)
        
    def close(self):
        if self.writer:
            self.writer.release()