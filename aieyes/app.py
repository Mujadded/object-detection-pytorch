from dataclasses import dataclass
from enum import Enum
import cv2
import torch
from PIL import Image
from concurrent.futures import ThreadPoolExecutor
from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer
import time
from typing import Any
from tesserocr import PyTessBaseAPI
from imutils.video import VideoStream
from imutils.video import FPS
import imutils
import argparse


@dataclass
class ObjectDetectionModel:
    
    name : str = "yolov5s"
    path : str = "ultralytics/yolov5"
    model : Any = torch.hub.load(path, name)
    
    def __call__(self,data):
        output = self.detect(data)
        return output
            
    def detect(self, data):
        result = self.model(data)
        result.render()
        return data

@dataclass
class ImageCaptioningModel:
    
    name: str = None
    path: str = "nlpconnect/vit-gpt2-image-captioning"
    preprocessor: Any = ViTImageProcessor.from_pretrained(path)
    model : Any = VisionEncoderDecoderModel.from_pretrained(path)
    postprocessor: Any = AutoTokenizer.from_pretrained(path)
    
    def __call__(self,data):
        preprocessed_data = self.preprocess(data)
        model_output = self.generate_caption(preprocessed_data)
        result = self.postprocess(model_output)
        return result
    
    def generate_caption(self, data):
        return self.model.generate(data)
    
    def preprocess(self, data):
        return self.preprocessor(data, return_tensors='pt').pixel_values
    
    def postprocess(self, data):
        output = self.postprocessor.batch_decode(data, skip_special_tokens=True)
        output = [word.strip() for word in output]
        return output
    
    
class OCRModel:
    name : str = 'tesseract-ocr'
        
    def __call__(self, data):
        with PyTessBaseAPI() as api:
            api.SetImage(Image.fromarray(data))
            text = api.GetUTF8Text()
        # print(api.AllWordConfidences())
        return text
    
class ModelType(Enum):
    OBJECT_DETECTION = 'object-detection'
    IMAGE_CAPTIONING = 'image-captioning'
    TEXT_READER = 'ocr'

class ModelFactory:
    
    def build(self, model_type):
        if model_type == ModelType.OBJECT_DETECTION:
            return ObjectDetectionModel()
        
        if model_type == ModelType.IMAGE_CAPTIONING:    
            return ImageCaptioningModel(name='ViT-gpt2')
        
        if model_type == ModelType.TEXT_READER:
            return OCRModel()
    
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

@dataclass    
class WebCamReader:
        
    def initialize(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.video_stream = VideoStream(src=0).start()
        time.sleep(2.0)
        self.fps = FPS().start()
        self.height= 1152
        self.width= 864
    
    def next_frame(self):
        frame = self.video_stream.read()
        frame = imutils.resize(frame, width=self.width, height=self.height)
        return frame

    def is_running(self):
        return True
    
    def close(self):
        self.fps.stop()
        cv2.destroyAllWindows()
        self.video_stream.stop()
        
    def get_video_width(self):
        return self.width
    
    def get_video_height(self):
        return self.height

def process_queue(futures, name):
    text = ''
    for i in range(len(futures)):
            future, timestamp = futures[i]
            if future.done():
            
                preds = future.result()
                if isinstance(preds, list):
                    text = preds[0]
                else:
                    text = preds
                print(f'{name} -> {preds}')

                futures.pop(i)
                break  # Exit the loop to avoid modifying the list while iterating over it
    return text

def update_text(text, futures, name):
    response = process_queue(futures, name)
    if response == '':
        return text
    return response

def add_text_to_image(text, image, yloc):
    cv2.putText(image, text, (0, yloc), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

def async_submit(executor, func, param, queue):
    future = executor.submit(func, param)
    queue.append((future, time.time()))

def show(frame, title='frame'):
    cv2.imshow(title, frame)

def interrupted():
    return cv2.waitKey(1) & 0xFF == ord('q')

def load_image(path):
    return cv2.imread(path)

def save_image(image, path):
    cv2.imwrite(path, image)
    
if __name__ == "__main__":

    # Create an argument parser
    parser = argparse.ArgumentParser(description='Process some images or videos.')

    # Add the --image and --video options
    parser.add_argument('--image', action='store_true', help='process an image')
    parser.add_argument('--video', action='store_true', help='process a video')
    parser.add_argument('--webcam', action='store_true', help='process a video directly from webcam')

    # Add the source and save path arguments
    parser.add_argument('--source_path', required=False, help='path to the source image or video')
    parser.add_argument('--save_path', required=False, help='path to save the processed image or video')

    # Parse the command line arguments
    args = parser.parse_args()
    
    factory = ModelFactory()
    object_detector = factory.build( ModelType.OBJECT_DETECTION )
    scene_descriptor = factory.build( ModelType.IMAGE_CAPTIONING )
    text_reader = factory.build(ModelType.TEXT_READER)
    
    def describe_scene(image):
        return scene_descriptor(image)

    def read_text(image):
        return text_reader(image)
    
    if args.image:
        image = load_image(args.source_path)
        annotated_image = object_detector(image.copy())
        desc = scene_descriptor(image)
        text = text_reader(image)
        
        print(f'Description: {desc}')
        print(f'OCR Result: {text}')
        
        add_text_to_image(text='description: ' + desc[0], image=annotated_image, yloc=image.shape[0]-20)
        add_text_to_image(text='ocr-text: ' + text, image=annotated_image, yloc=20)
        
        save_image(annotated_image, args.save_path)
          
    elif args.video or args.webcam:
        executor = ThreadPoolExecutor(max_workers=2)
        desc_futures, text_futures = [], []
        
        if args.video:
            video = VideoReader(args.source_path)                        
            video_writer = VideoWriter(args.save_path)
            video_writer.initialize(width=video.get_video_width(), height=video.get_video_height()) 
        elif args.webcam:
            video = WebCamReader()

        video.initialize() 
        
        desc_text = ''
        ocr_text = ''
        interval = 25
        frame_counter = 0
        
        while video.is_running():
            frame = video.next_frame()
            frame_counter += 1
            
            if frame is None:
                break
            
            annotated_frame = object_detector(frame.copy())
            
            if frame_counter % interval == 0:
                async_submit(executor, describe_scene, frame, desc_futures)
                async_submit(executor, read_text, frame, text_futures)
            
            desc_text = update_text( desc_text, desc_futures, 'captions')
            ocr_text = update_text( ocr_text, text_futures, 'ocr')
            
            add_text_to_image(text='description: ' + desc_text, image=annotated_frame, yloc=frame.shape[0]-20)
            add_text_to_image(text='ocr-text: ' + ocr_text, image=annotated_frame, yloc=20)
            
            show(annotated_frame)
            
            if interrupted():
                break
        
                
        video.close()
        if args.video:
            video_writer.close()
        cv2.destroyAllWindows()
    