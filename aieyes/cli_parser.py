import argparse

def parse_args():
    # Create an argument parser
    parser = argparse.ArgumentParser(description='Process some images or videos.')

    # Add the --image and --video options
    parser.add_argument('-i', '--image', action='store_true', help='process an image')
    parser.add_argument('-v', '--video', action='store_true', help='process a video')
    parser.add_argument('-c', '--camera', action='store_true', help='process live webcam feed')
    parser.add_argument('-s', '--save', action='store_true', help='save output')

    # Add the source and save path arguments
    parser.add_argument('source_path', help='path to the source image or video', nargs='?')
    parser.add_argument('save_path', help='path to save the processed image or video', nargs='?')
    parser.add_argument('-ww', '--width', help='window width if using --camera')
    parser.add_argument('-wh', '--height', help='window height if using --camera')

    # Parse the command line arguments
    args = parser.parse_args()
    
    if args.camera:
        args.save_path = args.source_path
            
    args_dict = vars(args)
    args_dict['window_size'] = (int(args.width), int(args.height)) if args.width and args.height else None
    return args_dict