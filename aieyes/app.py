import cv2
import cli_parser
from misc import *
from dataclasses import dataclass
from video_utils import VideoReader, VideoWriter
from concurrent.futures import ThreadPoolExecutor
from vision_models import ModelType, ModelFactory
    
def process_video(source, save_path=None, window_size=None):
    
    def describe_scene(image):
        return scene_descriptor(image)

    def read_text(image):
        return text_reader(image)
    
    executor = ThreadPoolExecutor(max_workers=2)
    desc_futures, text_futures = [], []
    
    video = VideoReader(source)   
    video.initialize()
        
    video_writer = None
    if save_path:
        video_writer = VideoWriter(save_path)
        # TODO: save video to using window_size dims?
        video_writer.initialize(width=video.get_video_width(), height=video.get_video_height())
    
    desc_text, ocr_text = '', ''
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
        
        show(annotated_frame, dims=window_size)
                
        if save_path:
            video_writer.write(annotated_frame)

        if interrupted():
            break
    
    video.close()
    if video_writer:
        video_writer.close()
    cv2.destroyAllWindows()
    
def process_image(source, save_path=None):
    image = load_image(source)
    annotated_image = object_detector(image.copy())
    desc = scene_descriptor(image)
    text = text_reader(image)
    
    add_text_to_image(text='description: ' + desc[0], image=annotated_image, yloc=image.shape[0]-20)
    add_text_to_image(text='ocr-text: ' + text, image=annotated_image, yloc=20)
    
    show(annotated_image, wait=True)
    if save_path:
        save_image(annotated_image, save_path)

if __name__ == "__main__":

    args = cli_parser.parse_args()
    
    factory = ModelFactory()
    object_detector = factory.build( ModelType.OBJECT_DETECTION )
    scene_descriptor = factory.build( ModelType.IMAGE_CAPTIONING )
    text_reader = factory.build(ModelType.TEXT_READER)
    
    if args['image']:    
        process_image(args['source_path'], args['save_path'])
    elif args['video']:
        process_video(args['source_path'], args['save_path'])  
    elif args['camera']:
        process_video(0, args['save_path'], args['window_size'])
            