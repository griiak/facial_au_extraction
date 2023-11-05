"""
This script located the apex frame for each trial based on 10
facial regions of interest.
"""
import glob
import os
import re
import sys

import numpy as np
from tqdm import tqdm

from smic_processing import find_apex_frame_of_clip

ROOT_FOLDER = f""
FRAME_DIFFS = f"{ROOT_FOLDER}\ROI_Diffs"
FRAMES_PATH = f"{ROOT_FOLDER}\Extracted_Frames"

THETA = 6
TAU = 30

txt_path = f"{ROOT_FOLDER}\Apex_frame_data.txt"
spk_path = f"{ROOT_FOLDER}\Apex_spike_data.txt"

def main():

    if not os.path.exists(FRAME_DIFFS):
        os.makedirs(FRAME_DIFFS)

    # Loop through all experiments
    experiments = glob.glob(FRAMES_PATH + "\*/")
    print("experiments: ", experiments)
    for experiment in experiments:
        experiment_name = os.path.basename(os.path.normpath(experiment))
        print(experiment_name)
        if not os.path.exists(f"{FRAME_DIFFS}\{experiment_name}"):
            os.makedirs(f"{FRAME_DIFFS}\{experiment_name}")

        # Loop through all blocks
        blocks = glob.glob(experiment + "*/")
        for block in tqdm(blocks, total=3):
            block_name = os.path.basename(os.path.normpath(block))
            block_num = int(block_name.split("_")[-1])

            # Sort trial paths in ascending number order
            # (instead of alphabetically)
            trials = glob.glob(block + "*/")
            trials2 = [os.path.basename(os.path.normpath(t)) for t in trials]
            trials2 = sorted(trials2, key=lambda x: int(re.findall(r"\d+", x)[0]))
            trials = [block + t + "\\" for t in trials2]

            # Loop through all trials
            for trial in trials:
                trial_name = os.path.basename(os.path.normpath(trial))
                trial_num = int(trial_name.split("_")[-1])
                print(experiment_name, block_num, trial_num)
                feats_path = f"{FRAME_DIFFS}\{experiment_name}\Block_{block_num}_Trial_{trial_num}.npy"

                if os.path.exists(feats_path):
                    continue

                # Sort frame paths in ascending number order
                frame_paths = glob.glob(trial + "*.jpg")
                frame_paths2 = [os.path.basename(os.path.normpath(t)) for t in frame_paths]
                frame_paths2 = sorted(
                    frame_paths2, key=lambda x: int(re.findall(r"\d+", x)[0])
                )
                frame_paths = [trial + t for t in frame_paths2]

                # Return pixel intesity difference array in ROIs.
                # The apex frame is the one with the max difference.
                features = find_apex_frame_of_clip(frame_paths)
                if features is None:
                    continue
                with open(
                    f"{FRAME_DIFFS}\{experiment_name}\Block_{block_num}_Trial_{trial_num}.npy",
                    "wb",
                ) as f:
                    np.save(f, features)


if __name__ == '__main__':
    sys.exit(main())  