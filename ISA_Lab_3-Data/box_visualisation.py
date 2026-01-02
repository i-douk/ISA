import os
import cv2
from tqdm import tqdm
import pickle


def image_with_bounding_boxes(img, boxes):
    ''' Adds visualisation of predicted bounding boxes to a single frame.

    :param img: A single frame from a video sequence.
    :param boxes: List of bounding box predictions.
    :return: The input frame with a visualisation of the predicted bounding boxes.
    '''
    color = (0, 0, 255)
    thickness = 1
    for box in boxes:
        start_point = (box[0], box[1])
        end_point = (box[2], box[3])
        img = cv2.rectangle(img, start_point, end_point, color, thickness)
    return img


# Define paths for in- and outputs
image_sequence_path = '../sequence'    # <--- TODO: Change this path to the folder the image sequence is stored in!
predictions_file_path = 'predictions.pickle'
video_output_path = 'video_output.mp4'

number_of_frames = len(os.listdir(image_sequence_path))
images_names_list = [os.path.join(image_sequence_path, f'S1-T1-C.{i:05d}.jpeg') for i in range(number_of_frames)]

# Load bounding box predictions from file
with open(predictions_file_path, 'rb') as f:
    predictions = pickle.load(f)

# Initialise output video
first_frame_path = images_names_list[0]
first_frame = cv2.imread(first_frame_path)
height, width, layers = first_frame.shape
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
video = cv2.VideoWriter(video_output_path, fourcc, 20, (width, height))

# Add visualisation of bounding box predictions to every frame
for frame_count in tqdm(range(0, number_of_frames)):
    image_path = images_names_list[frame_count]
    frame_path = image_path
    frame = cv2.imread(frame_path)
    pred_image = image_with_bounding_boxes(frame, predictions[frame_count])
    video.write(pred_image)
cv2.destroyAllWindows()
video.release()