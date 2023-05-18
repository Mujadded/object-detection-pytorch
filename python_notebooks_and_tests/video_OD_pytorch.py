from torchvision.models import detection
from torchvision.io import read_image,read_file
import numpy as np
import torch
import torchvision.transforms as transforms
import time

import matplotlib.pyplot as plt
import torchvision.transforms.functional as F
plt.rcParams["savefig.bbox"] = "tight"


def plot(imgs, **imshow_kwargs):
    if not isinstance(imgs[0], list):
        # Make a 2d grid even if there's just 1 row
        imgs = [imgs]

    num_rows = len(imgs)
    num_cols = len(imgs[0])
    _, axs = plt.subplots(nrows=num_rows, ncols=num_cols, squeeze=False)
    for row_idx, row in enumerate(imgs):
        for col_idx, img in enumerate(row):
            ax = axs[row_idx, col_idx]
            img = F.to_pil_image(img.to("cpu"))
            ax.imshow(np.asarray(img), **imshow_kwargs)
            ax.set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])

    plt.tight_layout()

classes=[]
with open("coco_classlabels.txt") as f:
    classes=[line.strip() for line in f.readlines()]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CLASSES = classes
COLORS = np.random.randint(0, 255, size=(len(CLASSES), 3))
MIN_CONFIDENCE= 0.70
COLORS = [tuple(color) for color in COLORS]
model = detection.fasterrcnn_resnet50_fpn(pretrained=True, progress=True,
	num_classes=len(CLASSES), pretrained_backbone=True).to(DEVICE)

model.eval()
from torchvision.io import read_video

frames, _, _ = read_video('./sample.mp4', output_format="TCHW")
from torchvision.utils import draw_bounding_boxes
new_frames = torch.zeros_like(frames)
frame_idx = 0
for frame in frames:
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor()
    ])

    transformed_image = transform(frame)

    transformed_image = transformed_image.unsqueeze(0)

    transformed_image = transformed_image.to(DEVICE)
    detections = model(transformed_image)[0]

    idx = detections['scores'] > 0.90
    labels = [f'{CLASSES[x-1].capitalize()}: {detections["scores"][x]*100:.2f}%' for x in detections['labels'][idx]]

    box_image=draw_bounding_boxes(frame,detections['boxes'][idx],labels=labels,width=4,font_size=25,font='arial', colors=COLORS)

    new_frames[frame_idx] = box_image
    frame_idx += 1

from torchvision.io import write_video

write_video('./output.mp4',new_frames.permute(0,2,3,1),fps=30)