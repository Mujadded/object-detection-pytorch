# AI Eyes

AIEyes - An AI powered visual aid for the visually impared.

## Requirements

- Python 3.10.8
- pip

## Installation

Install the project dependencies using pip:

`pip install -r requirements.txt`

Note: The project was tested on MacOS. To run `tesserocr` tesseract wrapper on windows, please follow instructions [here](https://github.com/sirfz/tesserocr)

Note: yolov5 project was cloned from [here](https://github.com/ultralytics/yolov5)

## Usage

Run the project using the following commands, to run the system on

A single image:

`python app.py --image <source.jpg>` (display only)
`python app.py --image --save <source.jpg> <destination.jpg>` (display and save result)

Video file:

`python app.py --video <source.mp4>` (display only)
`python app.py --video --save <source.mp4> <destination.avi>` (display and save result)

Webcam:

`python app.py --camera` (display only)
`python app.py --camera --save <destination.avi>` (display and save result)
`python app.py --camera --width <integer> --height <integer>` (display with specified window size)

 