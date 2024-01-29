import torch
from PIL import Image
from enum import Enum
from typing import Any
from dataclasses import dataclass
from tesserocr import PyTessBaseAPI
from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer, AutoProcessor, AutoModelForCausalLM

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

    def __post_init__(self):
        if(self.name == 'GIT-ms'):
            self.path = "microsoft/git-base-coco"
            self.preprocessor = AutoProcessor.from_pretrained(self.path)
            self.model = AutoModelForCausalLM.from_pretrained(self.path)
            self.postprocessor = self.preprocessor
        elif(self.name == 'ViT-gpt2'):
            self.path = "nlpconnect/vit-gpt2-image-captioning"
            self.preprocessor = ViTImageProcessor.from_pretrained(self.path)
            self.model = VisionEncoderDecoderModel.from_pretrained(self.path)
            self.postprocessor = AutoTokenizer.from_pretrained(self.path)
    
    def __call__(self,data):
        preprocessed_data = self.preprocess(data)
        model_output = self.generate_caption(preprocessed_data)
        result = self.postprocess(model_output)
        return result
    
    def generate_caption(self, data):
        if self.name == 'ViT-gpt2':
            caption = self.model.generate(data)
        elif self.name == 'GIT-ms':
            caption = self.model.generate(pixel_values=data, max_length=50)
        return caption
    
    def preprocess(self, data):
        if self.name == 'ViT-gpt2':
            pixel_values = self.preprocessor(data, return_tensors='pt').pixel_values
        elif self.name == 'GIT-ms':
            pixel_values = self.preprocessor(images=data, return_tensors='pt').pixel_values
        return pixel_values
    
    def postprocess(self, data):
        output = self.postprocessor.batch_decode(data, skip_special_tokens=True)
        if self.name == 'ViT-gpt2':
            output = [word.strip() for word in output]
        elif self.name == 'GIT-ms':
            output = output[0]
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
    