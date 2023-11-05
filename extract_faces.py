"""
This script extracts the subject's face (centered on nose tip)
from the base and apex frames of each trial.
"""

import os
import sys

import cv2
import face_recognition
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

ROOT_FOLDER = ""
FRAMES_PATH = f"{ROOT_FOLDER}\Extracted_Frames"
SAVE_FOLDER = f"{ROOT_FOLDER}\Extracted_Faces"
APEX_INFO_PATH = f"{ROOT_FOLDER}\Apex_frame_data.txt"

# Return face image coordinates centering on nose tip
def locate_face(image, frame_path):
    face_locations = face_recognition.face_locations(
        image, number_of_times_to_upsample=0
    )

    max_face = 0
    max_size = -1

    # Find all faces, return the biggest one
    for idx, face_location in enumerate(face_locations):
        top, right, bottom, left = face_location
        face_size = (bottom - top) * (right - left)
        if face_size > max_size:
            max_face = idx
            max_size = face_size

    top, right, bottom, left = face_locations[max_face]
    face_landmarks = face_recognition.face_landmarks(
        image, face_locations=face_locations, model="small"
    )
    nose_tip = face_landmarks[max_face]["nose_tip"][0]

    # Nose located closer to the top of image, centered left-to-right
    face_image = image[
        nose_tip[1] - 456 : nose_tip[1] + 256, nose_tip[0] - 356 : nose_tip[0] + 356
    ]
    return (
        face_image,
        nose_tip,
        (nose_tip[1] - 456, nose_tip[0] + 356, nose_tip[1] + 256, nose_tip[0] - 356),
    )


def extract_face(apex_frame_path, base_frame_path, save_path):

    # Extract fase from clip's base frame
    base_img = cv2.imread(base_frame_path)
    base_face, base_nose, base_face_loc = locate_face(base_img, base_frame_path)
    cv2.imwrite(save_path + "base_frame.jpg", base_face)


    apex_img = cv2.imread(apex_frame_path)

    # Center face image of apex frame on exact same position as in
    # base frame.
    _, apex_nose, apex_face_loc = locate_face(apex_img, apex_frame_path)
    apex_diffs = [
        base_face_loc[0] - base_nose[1],
        base_face_loc[1] - base_nose[0],
        base_face_loc[2] - base_nose[1],
        base_face_loc[3] - base_nose[0],
    ]
    (b_top, b_right, b_bottom, b_left) = [
        apex_nose[1] - (base_nose[1] - base_face_loc[0]),
        apex_nose[0] - (base_nose[0] - base_face_loc[1]),
        apex_nose[1] - (base_nose[1] - base_face_loc[2]),
        apex_nose[0] - (base_nose[0] - base_face_loc[3]),
    ]
    face = apex_img[b_top:b_bottom, b_left:b_right]
    cv2.imwrite(save_path + "apex_frame.jpg", apex_face)
    return



def main():

    if not os.path.exists(SAVE_FOLDER):
        os.makedirs(SAVE_FOLDER)

    apex_df = pd.read_csv(APEX_INFO_PATH, sep="\s+", header=0)
    missing_counter = 0

    # Loop through all trials included in Apex_frame_data
    for index, row in tqdm(apex_df.iterrows(), total=len(apex_df)):
        experiment = row["Experiment"]
        if not os.path.exists(f"{SAVE_FOLDER}\{experiment}\\"):
            os.makedirs(f"{SAVE_FOLDER}\{experiment}\\")

        block = row["Block"]
        if not os.path.exists(f"{SAVE_FOLDER}\{experiment}\Block_{block}\\"):
            os.makedirs(f"{SAVE_FOLDER}\{experiment}\Block_{block}\\")

        trial = row["Trial"]

        # Check if trial has already been processed
        if not os.path.exists(f"{SAVE_FOLDER}\{experiment}\Block_{block}\Trial_{trial}\\"):
            os.makedirs(f"{SAVE_FOLDER}\{experiment}\Block_{block}\Trial_{trial}\\")
        else:
            continue

        apex_frame = row["Apex_Frame"]
        base_frame = row["Start_Frame"]

        apex_frame_path = f"{FRAMES_PATH}\{experiment}\Block_{block}\Trial_{trial}\\frame_{apex_frame}.jpg"
        base_frame_path = f"{FRAMES_PATH}\{experiment}\Block_{block}\Trial_{trial}\\frame_{base_frame}.jpg"
        print(experiment, block, trial)

        # If either frame is missing, skip this trial and add to error counter
        if (os.path.exists(apex_frame_path)) and (os.path.exists(base_frame_path)):
            extract_face(
                apex_frame_path=apex_frame_path,
                base_frame_path=base_frame_path,
                save_path=f"{SAVE_FOLDER}\{experiment}\Block_{block}\Trial_{trial}\\",
            )
        else:
            missing_counter += 1
    print(missing_counter, " trials were discarded due to missing frames.")


if __name__ == '__main__':
    sys.exit(main())  