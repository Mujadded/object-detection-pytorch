import cv2 
import time

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

def show(frame, title='frame', wait=False, dims=None):
    if dims:
        frame = cv2.resize(frame, dims)
    cv2.imshow(title, frame)

    if wait:
        cv2.waitKey(0)
        cv2.destroyAllWindows()

def interrupted():
    return cv2.waitKey(1) & 0xFF == ord('q')

def load_image(path):
    return cv2.imread(path)

def save_image(image, path):
    cv2.imwrite(path, image)