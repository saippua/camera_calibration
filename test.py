#! /usr/bin/python
import numpy as np
from time import sleep
import cv2
import argparse

CSI_MODES = {
        1: { "width": 1280, "height": 720, "rate": 60 },
        2: { "width": 1920, "height": 1080, "rate": 30 },
        3: { "width": 1640, "height": 1232, "rate": 30 },
        4: { "width": 3280, "height": 1848, "rate": 28 },
        5: { "width": 3280, "height": 2464, "rate": 21 }
}
CSI_ID = 0


def create_csi_config(id, mode):
    return f"nvarguscamerasrc sensor-id={id} ! " + \
            "video/x-raw(memory:NVMM), " + \
           f"width=(int){mode['width']}, " + \
           f"height=(int){mode['height']}, " + \
           f"framerate=(fraction){mode['rate']}/1, format=(string)NV12 ! " + \
            "nvvidconv ! video/x-raw, format=(string)BGRx ! " + \
            "videoconvert ! video/x-raw, format=(string)BGR ! " + \
            "appsink";

def t265_callback(frame):
    global image, t265_stamp, t265_started, new_frame
    if frame.is_frameset():
        frameset = frame.as_frameset()
        frame = frameset.get_fisheye_frame(args.camera + 1).as_video_frame()

        ts = frameset.get_timestamp()

        arr = np.asanyarray(frame.get_data())

        frame_mutex.acquire()
        image = arr
        t265_stamp = ts
        new_frame = True
        frame_mutex.release()

        t265_started = True


def unpack_intrinsics(data):
    K = data['camera_matrix']
    D = data['distortion_coefficients']
    w, h = data['camera_resolution']


    D = D[:4]

    return K, D, w, h


class INT_CAM0: # fisheye rad-tan
    fx=286.2683105
    fy=286.29598999
    cx=418.566589
    cy=403.1160888
    k1=-0.00476462999358773
    k2=0.0415727309882641
    p1=-0.038710381835699
    p2=0.006511746905744
    w, h = 848, 800

    K = np.array([[fx, 0, cx],[0, fy, cy], [0,0,1]])
    D = np.array([k1, k2, p1, p2])

class INT_CAM1: # fisheye rad-tan

    fx=286.0837097
    cx=414.3789978
    fy=286.028686523
    cy=401.09240722
    k1=-0.00564806396141648
    k2=0.0425024405121803
    p1=-0.0404532216489315
    p2=0.0072439508512616
    w, h = 848, 800

    K = np.array([[fx, 0, cx],[0, fy, cy], [0,0,1]])
    D = np.array([k1, k2, p1, p2])

class INT_CAM2: # omnidirectional rad-tan
    fx=1157.7581937
    fy=1154.2590098
    cx=788.38097880
    cy=690.68415817
    k1=1.5132844070
    k2=-1.770286235
    p1=-0.000514584
    p2=0.0039635525
    xi=1.8930205493
    w, h = 1640, 1232

    K = np.array([[fx, 0, cx],[0, fy, cy], [0,0,1]])
    D = np.array([k1, k2, p1, p2])
    XI = np.array([xi])


def undistort_fisheye(img, intr, fov=120):
    f = np.abs(intr.w/(2*np.tan(np.deg2rad(fov/2))))
    K_new = np.array([[f, 0, intr.w/2],[0, f, intr.h/2],[0,0,1]])
    img = cv2.fisheye.undistortImage(img, intr.K, intr.D, None, K_new)
    return img


def undistort_omni(img, intr, fov=120):
    f = np.abs(intr.w/(2*np.tan(np.deg2rad(fov/2))))
    K_new = np.array([[f, 0, intr.w/2],[0, f, intr.h/2],[0,0,1]])
    img = cv2.omnidir.undistortImage(img, intr.K, intr.D, intr.XI, cv2.omnidir.RECTIFY_PERSPECTIVE, None, K_new)
    return img



if __name__ == '__main__':

    parser = argparse.ArgumentParser(
            prog="Monsterborg Camera Calibration Tester",
            description="Tests calibration by undistorting cameras.")

    parser.add_argument('-m', '--csi_mode', type=int, choices=range(1,6), default=3)
    parser.add_argument('-c', '--camera', type=int, choices=range(3), default=2)

    args = parser.parse_args()

    np.set_printoptions(suppress=True)

    t265 = args.camera < 2
    csi  = args.camera == 2

    if t265:
        import pyrealsense2.pyrealsense2 as rs
        from threading import Lock

        t265_stamp = 0
        t265_started = False
        new_frame = False
        frame_mutex = Lock()

        pipe = rs.pipeline()
        cfg = rs.config()

        pipe.start(cfg, t265_callback)
        print("T265 started.")

    if csi:
        cap = cv2.VideoCapture(create_csi_config(CSI_ID, CSI_MODES[args.csi_mode]))
    try:
        while True:
            if t265 and t265_started and new_frame:

                frame_mutex.acquire()
                img = image.copy()
                new_frame = False
                frame_mutex.release()

                if args.camera == 0:
                    undistorted_img = undistort_fisheye(img, INT_CAM0)
                if args.camera == 1:
                    undistorted_img = undistort_fisheye(img, INT_CAM1)

                cv2.imshow("t265", undistorted_img)

            if csi:
                retval, img = cap.read()
                if retval:
                    undistorted_img = undistort_omni(img, INT_CAM2)
                    cv2.imshow("csi", undistorted_img)
                else:
                    continue

            key = cv2.waitKey(1)
            if key == ord('q'):
                break
    except KeyboardInterrupt:
        pass


