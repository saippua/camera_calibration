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
image = None # store image from callback here.
image_other = None # store the other iamge from t265 here
t265_started = False
t265_stamp = 0
new_frame = False
previous_tick = 0
timed_capture_running = False

snapshot_save_start_index = 50
snapshots = []

TIMER_RATE = 4.
CHESSBOARD_SQUARES = (7,5)
# CHESSBOARD_SQUARES = (8,6)
# CHESSBOARD_SIZE = ((195/8)*7, (146/8)*7)
CHESSBOARD_SIZE = (146., 97.1)
# CHESSBOARD_SIZE = (195, 146)
calibration_flags = None # cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC+cv2.fisheye.CALIB_FIX_SKEW
# subpixel refinement criteria
def CRITERIA(amount):
    return (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, amount)

CSI_MODES = {
        1: { "width": 1280, "height": 720, "rate": 60 },
        2: { "width": 1920, "height": 1080, "rate": 30 },
        3: { "width": 1640, "height": 1232, "rate": 30 },
        4: { "width": 3280, "height": 1848, "rate": 28 },
        5: { "width": 3280, "height": 2464, "rate": 21 }
}
CSI_ID = 0

headless = False

def tiles(imgs, rows, cols, scale=1.):
    N, H, W, C = imgs.shape
    H = int(H*scale)
    W = int(W*scale)


    img = np.zeros((rows*H, cols*W, C), dtype=np.uint8)
    i = 0
    for row in range(rows):
        for col in range(cols):
            y = row*H; x = col*W;
            if i < N:
                img[y:y+H,x:x+W,:] = cv2.resize(imgs[i], (W,H), interpolation=cv2.INTER_AREA)
            i += 1
    return img

def create_csi_config(id, mode):
    return f"nvarguscamerasrc sensor-id={id} ! " + \
            "video/x-raw(memory:NVMM), " + \
           f"width=(int){mode['width']}, " + \
           f"height=(int){mode['height']}, " + \
           f"framerate=(fraction){mode['rate']}/1, format=(string)NV12 ! " + \
            "nvvidconv ! video/x-raw, format=(string)BGRx ! " + \
            "videoconvert ! video/x-raw, format=(string)BGR ! " + \
            "appsink";

def calculate_grid_points(num_squares, physical_size):
    """
    Calculates 3D points from chessboard properties.
    `num_squares`: (w, h)
    `physical_size`: (w, h) in mm.
    """
    num_squares = np.array(num_squares)
    physical_size = np.array(physical_size)

    square_dims = (num_squares / physical_size)
    grid_points = np.zeros((num_squares[0]*num_squares[1],1,3), np.float32)
    grid_points[:,0,:2] = np.mgrid[0:num_squares[0],0:num_squares[1]].T.reshape(-1,2)
    grid_points[:,0,:2] *= num_squares

    return grid_points

def detect_board(img):

    ds_ratio = 1 # Downscale ratio for speed.
    ann_ratio = 1

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray_downscaled = cv2.resize(gray, (gray.shape[1]//ds_ratio, gray.shape[0]//ds_ratio), interpolation=cv2.INTER_AREA)
    ret, corners = cv2.findChessboardCorners(gray_downscaled, np.array(CHESSBOARD_SQUARES), cv2.CALIB_CB_FAST_CHECK)
    img_annotated = cv2.resize(img, (img.shape[1]//ann_ratio, img.shape[0]//ann_ratio), interpolation=cv2.INTER_AREA)

    if ret:
        corners *= ds_ratio
        corners_refined = cv2.cornerSubPix(gray, corners, (10,10), (-1,-1), CRITERIA(0.01))
        # corners_refined = corners
        cv2.drawChessboardCorners(img_annotated, CHESSBOARD_SQUARES, corners_refined/ann_ratio, ret)
    else:
        corners_refined = None
        cv2.line(img_annotated, (0,0), img_annotated.shape[:2][::-1], (0,0,255), 50)

    # return gray
    return corners_refined, img_annotated

def calibrate(snapshots, fisheye=False):
    pts3d = calculate_grid_points(CHESSBOARD_SQUARES, CHESSBOARD_SIZE)

    imgs_annotated = []
    imgpoints = []
    h, w, c = snapshots[0].shape
    print("Image has ", c, "channels.")

    for img in tqdm(snapshots, desc="Finding chessboard corners"):
        if img is tuple:
            img = img[0]

        pts2d, img_annotated = detect_board(img)
        imgs_annotated.append(img_annotated)
        imgpoints.append(pts2d)

    # Draw tiles here.
    rows = 4
    tiled_preview = tiles(np.array(imgs_annotated), cols=((len(imgs_annotated)-1)//rows) + 1, rows=rows, scale=0.4)
    cv2.imshow("Preview", tiled_preview)
    cv2.waitKey(0)

    if args.validate:
        for img in imgs_annotated:
            cv2.imshow("Preview", img)
            cv2.waitKey(0)

    cv2.destroyAllWindows()

    objpoints = [ pts3d for i in range(len(imgpoints)) if imgpoints[i] is not None ]
    imgpoints = [ pts2d for pts2d in imgpoints if pts2d is not None ]

    if fisheye:
        N_OK = len(imgpoints)
        K = np.zeros((3,3))
        D = np.zeros((4,1))
        rvecs = [np.zeros((1,1,3), dtype=np.float64) for i in range(N_OK)]
        tvecs = [np.zeros((1,1,3), dtype=np.float64) for i in range(N_OK)]
        ret, _, _, _, _ = \
                cv2.fisheye.calibrate(
                        objpoints, 
                        imgpoints, 
                        (w,h), 
                        K, 
                        D, 
                        rvecs, 
                        tvecs, 
                        calibration_flags, 
                        CRITERIA(1e-6))
        # ret, K, D, rvecs, tvecs = cv2.fisheye.calibrate(objpoints, imgpoints, (w, h), None, None)
    else:
        ret, K, D, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, (w, h), None, None)
        D = D.squeeze()

    fmt_string = f"cam_{args.camera}_%Y-%m-%d_%H-%M-%S.npz"
    filename = datetime.now().strftime(fmt_string)
    filename_latest = f"cam_{args.camera}_latest.npz"
    np.savez(filename, camera_matrix=K, distortion_coefficients=D, camera_resolution=(w,h))
    np.savez(filename_latest, camera_matrix=K, distortion_coefficients=D, camera_resolution=(w,h))
    
    print(K)
    print(D)

    print(f"Saved calibration to {filename}")

def t265_callback(frame):
    global image, image_other, t265_stamp, t265_started, new_frame
    if frame.is_frameset():
        frameset = frame.as_frameset()
        frame = frameset.get_fisheye_frame(args.camera + 1).as_video_frame()
        frame_other = frameset.get_fisheye_frame(2 - args.camera).as_video_frame()
        ts = frameset.get_timestamp()

        if t265_stamp != ts:
            t265_stamp = ts
            # ignore the first frame of a pair. 
            # We want the second frame where both images are new.
            return 

        img = np.asanyarray(frame.get_data())
        img_other = np.asanyarray(frame_other.get_data())

        frame_mutex.acquire()
        image = img
        image_other = img_other
        new_frame = True
        frame_mutex.release()

        t265_started = True

def save_snaps(snapshots):
    for i in range(len(snapshots)):
        if isinstance(snapshots[i], tuple):
            cv2.imwrite(f"./snapshots/{args.camera}_0_{i+snapshot_save_start_index}.png", snapshots[i][0])
            cv2.imwrite(f"./snapshots/{1 - args.camera}_0_{i+snapshot_save_start_index}.png", snapshots[i][1])
        else:
            cv2.imwrite(f"./snapshots/{args.camera}_{args.csi_mode}_{i+snapshot_save_start_index}.png", snapshots[i])

def load_snaps():
    snapshots = []
    snaps = os.listdir("./snapshots")
    snaps = sorted(snaps)

    for file in snaps:
        if not os.path.isfile(f"./snapshots/{file}"):
            continue
        cam_index = file.split('_')[0]

        if int(cam_index) == args.camera:
            if cam_index == 2:
                csi_mode = file.split('_')[1]
                if int(csi_mode) != args.csi_mode:
                    continue
            img = cv2.imread(f"./snapshots/{file}")
            snapshots.append(img)

    return snapshots

def cleanup():
    cv2.destroyAllWindows()
    if args.camera < 2:
        pipe.stop()
    else:
        cap.release()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
            prog="Monsterborg Camera Calibrator",
            description="Script for calibrating camera sensors on the monsterborg")

    parser.add_argument('--headless', action='store_true')
    parser.add_argument('-m', '--csi_mode', type=int, choices=range(1,6), default=2)
    parser.add_argument('-c', '--camera', type=int, choices=range(3))
    parser.add_argument('-l', '--load', action='store_true')
    parser.add_argument('-t', '--timer', action='store_true')
    parser.add_argument('-f', '--fisheye', action='store_true')
    parser.add_argument('-v', '--validate', action='store_true')
    

    args = parser.parse_args()

    use_fisheye = args.fisheye

    if args.load:
        snapshots = load_snaps()
        calibrate(snapshots, fisheye=use_fisheye)
        print("Done.")
        exit()

    if args.camera == 2:
        cap = cv2.VideoCapture(create_csi_config(CSI_ID, CSI_MODES[args.csi_mode]))
    elif args.camera < 2:
        import pyrealsense2.pyrealsense2 as rs
        pipe = rs.pipeline()
        cfg = rs.config()
        pipe.start(cfg, t265_callback)
        print("T265 started.")


    try:
        previous_tick = time.time()
        if args.timer:
            pb = tqdm(total = TIMER_RATE)

        while True:
            if args.camera < 2 and t265_started and new_frame:

                frame_mutex.acquire()
                img = image.copy()
                img_other = image_other.copy()
                new_frame = False
                frame_mutex.release()

                cv2.imshow("t265", img)
            elif args.camera < 2:
                continue # wait for camera to start.

            if args.camera == 2:
                retval, img = cap.read()
                if retval:
                    # img = detect_board(img)
                    cv2.imshow("csi", img)

            def select():
                global img, img_other
                if args.camera < 2:
                    img = np.stack([img, img, img], axis=2)
                    img_other = np.stack([img_other, img_other, img_other], axis=2)
                    snapshots.append((img, img_other))
                else:
                    snapshots.append(img)
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
                if delta > TIMER_RATE:
                    previous_tick = now
                    select()
                    pb.n = 0
                    pb.refresh()
                else:
                    pb.n = delta
                    pb.refresh()

            if key == ord('c'):
                save_snaps(snapshots)
                cleanup()
                calibrate(snapshots)
                break
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



