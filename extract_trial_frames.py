"""
This script loops through all experiment videos and extracts all
individual frames grouped by Experiment, Block and Trial number.
"""

import glob
import os
import sys

import cv2
import numpy as np
import pandas as pd
from ffmpeg_videostream import VideoStream
from tqdm import tqdm

ROOT_FOLDER = ""
VIDS_FOLDER = f"{ROOT_FOLDER}\SH10_Panasonic"
SAVE_FOLDER = f"{ROOT_FOLDER}\Face_Analysis\Extracted_Frames"
SCRIPT_FOLDER = f"{ROOT_FOLDER}\Scripts\Input_Data"

RESP_WINDOW = [0, 2] # Response time window (s)


def extract_clip_frames(vid_stream, resp_timestamp, fps, trial_dir):
    resp_window = resp_timestamp + RESP_WINDOW

    # Calculate first and last frame in response window.
    first_frame = int(np.floor(resp_window[0] * fps))
    last_frame = int(np.ceil(resp_window[1] * fps))

    # Set video stream limits.
    vid_stream.config(start_hms=resp_window[0], end_hms=resp_window[1])
    vid_stream.open_stream(showinfo=False)

    frame_num = first_frame

    while True:
        eof, frame = vid_stream.read()

        # Stop if video ends or response window end is reached
        if eof or frame_num > last_frame:
            break

        # Convert frame data to color image format
        arr = np.frombuffer(frame, np.uint8).reshape(
            int(vid_stream._shape[1] * 1.5), vid_stream._shape[0]
        )
        img = cv2.cvtColor(arr, cv2.COLOR_YUV2BGR_I420)

        save_path = f"{trial_dir}frame_{frame_num}.jpg"
        cv2.imwrite(save_path, img)
        frame_num += 1
        print(frame_num)


def main():

    if not os.path.exists(SAVE_FOLDER):
        os.makedirs(SAVE_FOLDER)

    # Get list of experiments from excel columns
    flashes_panasonic = pd.read_excel(f"{SCRIPT_FOLDER}\FullFlashFrames.xlsx", header=0)
    print(flashes_panasonic)
    experiments = list(flashes_panasonic.columns.values)

    # Loop through all experiments
    for experiment in experiments:
        if not os.path.exists(f"{SAVE_FOLDER}\{experiment}"):
            os.makedirs(f"{SAVE_FOLDER}\{experiment}")

        # Get relevant columns from the psychopy export
        export_path = glob.glob(f"{ROOT_FOLDER}\SH10_Psychopy\{experiment}*.csv")[0]
        export_df = pd.read_csv(export_path, header=0)
        export_df = export_df[
            [
                "task_1.thisTrialN",
                "category",
                "image_task.started",
            ]
        ]
        export_df = export_df[export_df["image_task.started"].notna()].reset_index(
            drop=True
        )
        print(f"Entering experiment {experiment}...")

        # Loop through all videos
        videos = glob.glob(f"{VIDS_FOLDER}\{experiment}*\*.MP4")
        for i in range(len(videos)):
            if flashes_panasonic[experiment][i] == 0:
                continue

            block_dir = f"{SAVE_FOLDER}\{experiment}\Block_{i+1}\\"
            txt_path = f"{block_dir}trials_info.txt"
            video = videos[i]

            if not os.path.exists(block_dir):
                os.makedirs(block_dir)

                # Write clip summary file for each block
                with open(txt_path, "w") as fh:
                    fh.write(f"Video path: {video}\n\n")
                    fh.write("Trial time_start time_end frame_start frame_end\n")

            print(f"\tExtracting frames from {video}")
            vid_stream = VideoStream(video)
            vid_fps = vid_stream._fps

            # Delay in seconds for each block
            camera_delay = flashes_panasonic[experiment][i] / vid_fps
            block_df = export_df[export_df["task_1.thisTrialN"] == i].reset_index(drop=True)
            # Eprime delay in seconds
            eprime_delay = block_df["image_task.started"].iloc[0]

            # Sync video delay to eprime delay
            RespTimestamps = block_df["image_task.started"] - eprime_delay + camera_delay

            # Loop through each trial clip and extract frames
            for j in tqdm(range(len(RespTimestamps))):
                trial_dir = f"{block_dir}Trial_{j+1}/"
                if os.path.exists(trial_dir):
                    continue
                os.makedirs(trial_dir)
                extract_clip_frames(
                    vid_stream=vid_stream,
                    resp_timestamp=RespTimestamps[j],
                    fps=vid_fps,
                    trial_dir=trial_dir,
                )
                resp_window = RespTimestamps[j] + RESP_WINDOW
                first_frame = int(np.floor(resp_window[0] * vid_fps))
                last_frame = int(np.ceil(resp_window[1] * vid_fps))

                # Add trial info to block summary
                with open(txt_path, "a") as fh:
                    fh.write(
                        f"{j+1} {resp_window[0]} {resp_window[1]} {first_frame} {last_frame}\n"
                    )

if __name__ == '__main__':
    sys.exit(main())  