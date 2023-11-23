# Camera Calibration Repository

This is a python codebase for tools for calibrating cameras.

The calibration tools in this toolbox were used in conjunction with [Kalibr](https://github.com/ethz-asl/kalibr) to calibrate cameras on the data collection testbed used in my master's thesis.
The testbed consists of a Realsense T265 camera and a waveshare IMX219-200 CSI camera. The T265 camera comes pre-calibrated with intrinsics and stereo extrinsics, however the CSI camera had to be calibrated for intrinsics and extrinsics w.r.t the realsense camera.

# Requirements

Required packages and the version I used (will probably work with other versions)

 - Ubuntu: 20.04
 - Docker: 24.0.7 (for Kalibr)
    - Kalibr: latest from repo (22/11/2023)
 - python: 3.8.10
    - opencv: 4.6.0
    - numpy: 1.23.1
    - tqdm: 4.66.1
    - aprilgrid: 0.3.0 (for multi-camera calibration only)


# Components

## calibrate.py

This is a calibration script for calibrating the cameras with OpenCV. Due to difficulties modeling the IMX219-200 lens topology this calibration script was not used. It is provided only for the record. Actual calibration of the cameras was instead performed using [Kalibr](https://github.com/ethz-asl/kalibr).

## collect.py

This script is used to collect frames. The camera to collect can be set using the flat `--camera <ID>` where `ID` is the index of the camera (0 for T265 left, 1 for T265 right, 2 for CSI camera. -1 can also be used to collect from all cameras simultaneously). Camera frames are saved in the `snapshots` directory with the following naming convention `<CAM_ID>_<CSI_MODE>_<FRAME_ID>_<TIMESTAMP_NS>.png`. If T265 camera is selected, then `CSI_MODE` will be 0. T265 frames use the realsense provided timestamp, and CSI frames use python timestamp at the time the frame is captured. `CSI_MODE` determines the framerate and resolution of the CSI camera and can be set with the flag `--csi_mode <CSI_MODE>`. For calibration we used mode `3`, which is 1640x1323 resolution at 30 FPS.

Additional flags for the collection script are:
 * `--timer <INTERVAL>` which specifies the interval used for automatic data capture
 * `--start <INDEX>` which allows setting a specific starting index for new data, for example if you wish to add to an existing dataset. Otherwise new frames will start at `FRAME_ID=0`.

### Usage
To use the collector, run the script with the desired flags. A preview window should open up showing live feed from one of the cameras. `SPACEBAR` can be used to select a frame. If using the timer (`--timer` flag), then spacebar will start or stop automatic collection, and a frame will be selected every `<INTERVAL>`.

After you are happy with the number of collected frames, press `s` which will save the frames to the `snapshots` folder. pressing `q` at any point during collection will quit the tool (WITHOUT saving any selected frames).

## test.py

This script is used to test single-camera calibration. Calibrated intrinsics are defined inside the script in classes `INT_CAM0`, `INT_CAM1` and `INT_CAM2`. The script assumes that realsense T265 uses the Kannala Brandt fisheye model and CSI camera uses the Omnidirectional camera model with radial and tangential distortion coefficients.

This script also supports the same `--camera` and `--csi_mode` flags as `collect.py`, with the exception that `CAM_ID` must be `0-2` (you can only test one camera at a time)

### Usage

Just run the script with the desired flags and a window will pop up showing undistorted frames from the selected camera.

# Calibration in Kalibr

Kalibr is not included in this repository, however I will lay out the calibration process here for posterity.

 1. Build the Kalibr docker container on a workstation. I was unable to build the container on Jetson Xavier, so I'm not sure if its possible to run on Tegra ARM. After building the container, you mount a folder and enter the container prompt (specific instructions for this are in Kalibr wiki section: [Installation](https://github.com/ethz-asl/kalibr/wiki/installation)).
 2. Kalibr requires data to be in rosbag format. To achieve this you must copy the collected frames from the snapshots folder to the container's data folder into a specific directory structure.  
    * The basic directory structure is this:  
    ```
    data/
    └── <dataset_name>
        ├── cam0
        ├── cam1
        └── cam2
    ```
    * Frames from each camera go into their respective folder. You only need folders for cameras you want to calibrate.
    * The filenames of the frames must also be modified so it's just the timestamp: `<TIMESTAMP_NS>.png`. This can be automated with a simple bash or python script.
 3. Once the data is structured correctly, you can run the kalibr bagcreater package: `rosrun kalibr kalibr_bagcreater --folder <path_to_dataset> --output-bag <path_to_output_bag>`. 
    * NOTE: if ros complains about not finding the package, make sure you run the setup script `catkin_ws/devel/setup.bash` as instructed in the kalibr documentation.
 4. You should now have the bagfile. You can check that it has the correct topics and each topic has the correct frames with the command `rosbag info <path_to_output_bag>`.
 5. To calibrate cameras, use the command  
 `rosrun kalibr kalibr_calibrate_cameras --bag `<path_to_output_bag>` --target <path_to_target_yaml> --models <model>-<distortion> --topics <bag_topics> --show-extraction`  
    * The target yaml defines the calibration target ([more info](https://github.com/ethz-asl/kalibr/wiki/calibration-targets)).
    * Choose a model and distortion that you think will work for the camera. You might have to try different ones ([more info](https://github.com/ethz-asl/kalibr/wiki/supported-models)).
    * You can choose multiple topics from the bag. Just define a model to use for each one.
    * I calibrated cameras in Kalibr one at a time. Doing them one at a time gives you more control over the individual frames (you don't have to worry about the target grid being fully visible in all cameras, which can be difficult to achieve).
    * If you calibrate many cameras at a time, I believe Kalibr will also calibrate extrinsics between the cameras. I didn't try this.


# Multi-camera calibration.

For multi-camera calibration the `multi_calibrate.py` script is provided.
The script operates on files in the snapshots folder, so first you must collect some frames from the cameras. You can only calibrate 2 cameras at a time. The cameras to use are defined by the `--camera1` and `--camera2` flags. You can also set the number of images to use for calibration with the flag `--n_imgs`. The default is 1, although I suggest to use more. Just make sure there are enough frames for both cameras in the snapshots folder. It's also important that the frames are synchornised, so frame id `N` from both cameras should have similar timestamps. They won't match exactly, but the difference should be like 0.1-0.3 seconds.

The script assumes that the target is an aprilgrid ([info](https://github.com/powei-lin/aprilgrid)). The properties of the april grid are defined in the python script under function `calculate_april_objpoints` variables `APRIL_DIM`, `TAG_SIZE` and `TAG_SPACE`.

The script reads camera intrinsics from numpy zip files (.npz). For the realsense cameras (cam0 and cam1) the npz file should have:

 * 'K': 3x3 intrinsic matrix
 * 'D': 4 distortion coefficients
 * 'w': image width
 * 'h': image height

For the CSI camera, it should have an additional key 'xi' which is the Xi parameter of the omnidirectional camera model. The naming convention of the files is `cam<CAM_ID>_intrinsics.npz`. `utils.py` has functions for loading and saving these npz files, so refer to those for more information. Intrinsics for all 3 cameras are provided in the repository.

When running the script, it first calculates apriltags from both images. Then it determines correspondences. This process can be previewed with the `--preview` flag which will visualize each frame as it is processed. Matching april tags are drawn in green and non-matching (only found in one image) in red. The matching april tags from camera 2 are also drawn on frame from camera 1 in blue.

After tag correspondences are detected, the extrinsics are calculated using OpenCV. Extrinsic components `R` and `t` are output to the terminal.


