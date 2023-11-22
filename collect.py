#! /usr/bin/python
import sys, os, time
import argparse
import cv2
import numpy as np
from threading import Lock
from tqdm import tqdm
from datetime import datetime

np.set_printoptions(suppress=True)

frame_mutex = Lock()
image0 = None
image1 = None
image2 = None
t265_started = False
t265_stamp = 0
new_frame = False
previous_tick = 0
timed_capture_running = False


snapshots = []

CSI_MODES = {
        1: { "width": 1280, "height": 720, "rate": 60 },
        2: { "width": 1920, "height": 1080, "rate": 30 },
        3: { "width": 1640, "height": 1232, "rate": 30 },
        4: { "width": 3280, "height": 1848, "rate": 28 },
        5: { "width": 3280, "height": 2464, "rate": 21 }
}
CSI_ID = 0

headless = False

def create_csi_config(id, mode):
    return f"nvarguscamerasrc sensor-id={id} ! " + \
            "video/x-raw(memory:NVMM), " + \
           f"width=(int){mode['width']}, " + \
           f"height=(int){mode['height']}, " + \
           f"framerate=(fraction){mode['rate']}/1, format=(string)NV12 ! " + \
            "nvvidconv ! video/x-raw, format=(string)BGRx ! " + \
            "videoconvert ! video/x-raw, format=(string)BGR ! " + \
            "appsink";

def save_snaps(snapshots):
    for i in range(len(snapshots)):
        if snapshots[i][0] is not None:
            cv2.imwrite(f"./snapshots/{0}_0_{str(i+args.start).zfill(3)}_{snapshots[i][3]}.png", snapshots[i][0])
        if snapshots[i][1] is not None:
            cv2.imwrite(f"./snapshots/{1}_0_{str(i+args.start).zfill(3)}_{snapshots[i][3]}.png", snapshots[i][1])
        if snapshots[i][2] is not None:
            cv2.imwrite(f"./snapshots/{2}_{args.csi_mode}_{str(i+args.start).zfill(3)}_{snapshots[i][4]}.png", snapshots[i][2])

def cleanup():
    cv2.destroyAllWindows()
    if t265:
        pipe.stop()
    if csi:
        cap.release()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
            prog="Monsterborg Camera Calibrator",
            description="Script for calibrating camera sensors on the monsterborg")

    parser.add_argument('--headless', action='store_true')
    parser.add_argument('-m', '--csi_mode', type=int, choices=range(1,6), default=3)
    parser.add_argument('-c', '--camera', type=int, choices=range(-1, 3), default=-1)
    parser.add_argument('-t', '--timer', type=float, default=0., help="Capture with timer, at given interval")
    parser.add_argument('-s', '--start', type=int, default=0, 
                        help="Start index of data collection (for saving if we want to keep earlier data)")

    args = parser.parse_args()

    t265 = args.camera < 2
    csi = args.camera == -1 or args.camera == 2

    if csi:
        cap = cv2.VideoCapture(create_csi_config(CSI_ID, CSI_MODES[args.csi_mode]))

    if t265:
        import pyrealsense2.pyrealsense2 as rs
        pipe = rs.pipeline()
        cfg = rs.config()
        cfg.disable_all_streams()
        cfg.enable_stream(rs.stream.fisheye, 1)
        cfg.enable_stream(rs.stream.fisheye, 2)

        # # possible fix to hanging on start
        # ctx = rs.context()
        # devices = ctx.query_devices()
        # for dev in devices:
        #     dev.hardware_reset()

        pipe.start(cfg)
        print("T265 started.")


    try:
        previous_tick = time.time()
        img0 = img1 = img2 = None
        ts0 = ts2 = None
        if args.timer:
            pb = tqdm(total = args.timer)

        while True:
            if t265:
                retval, frames = pipe.try_wait_for_frames()

                if retval:
                    frame0 = frames.get_fisheye_frame(1).as_video_frame()
                    frame1 = frames.get_fisheye_frame(2).as_video_frame()

                    ts0 = int(frames.get_timestamp() * 1e6)
                    img0 = np.asanyarray(frame0.get_data())
                    img1 = np.asanyarray(frame1.get_data())


            if csi:
                retval, cap_img = cap.read()
                if retval:
                    img2 = cap_img
                    ts2 = int(time.time() * 1e9)

            if csi:
                cv2.imshow("csi camera", img2)
            elif t265:
                cv2.imshow("t265 (left)", img0)

            def select():
                global img0, img1, img2, ts0, ts2
                
                if t265:
                    img0 = np.stack([img0, img0, img0], axis=2)
                    img1 = np.stack([img1, img1, img1], axis=2)
                snapshots.append((img0, img1, img2, ts0, ts2))

                print(f"Saved {len(snapshots)} snapshots.")

            key = cv2.waitKey(1)
            if key == ord(' '):
                if args.timer:
                    timed_capture_running ^= True
                    previous_tick = time.time()
                    pb.n = 0
                    pb.refresh()
                else:
                    select()

            if args.timer and timed_capture_running:
                now = time.time()
                delta = now - previous_tick
                if delta > args.timer:
                    previous_tick = now
                    select()
                    pb.n = 0
                    pb.refresh()
                else:
                    pb.n = delta
                    pb.refresh()

            if key == ord('s'):
                save_snaps(snapshots)
                cleanup()
                break
            if key == ord('q'):
                cleanup()
                break
    except KeyboardInterrupt:
        cleanup()
    finally:
        print("Exiting.")



