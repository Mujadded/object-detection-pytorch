import torch
from PIL import Image
from enum import Enum
from typing import Any
from dataclasses import dataclass
from tesserocr import PyTessBaseAPI
from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer
from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer

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
    
@dataclass 
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
    