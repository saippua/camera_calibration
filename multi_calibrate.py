#! /usr/bin/python
import argparse
from inspect import getmembers
import cv2
from aprilgrid import Detector
import numpy as np
np.set_printoptions(suppress=True)
import glob
from tqdm import tqdm


parser = argparse.ArgumentParser(
        prog="Monsterborg Multi-Camera Calibrator",
        description="Script for calibrating extrinsics between camera sensors on the monsterborg")

parser.add_argument('-c1', '--camera1', type=int, choices=range(3), default=0)
parser.add_argument('-c2', '--camera2', type=int, choices=range(3), default=2)
parser.add_argument('-n', '--n_imgs', type=int, default=1)
parser.add_argument('-p', '--preview', action='store_true')

args = parser.parse_args()

from utils import *

ad = Detector("t36h11")

preview=args.preview
N_img = args.n_imgs
cams = (args.camera1, args.camera2)

def detect_april(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    detections = ad.detect(gray)
    detections.sort(key=lambda x: x.tag_id)
    return detections

def visualize_april(img, detections, valid_tiles, alt_colors=False):

    for d in detections:
        midpoint = np.array([0.,0.])

        pts = np.moveaxis(d.corners, 0, 1)

        if d.tag_id in valid_tiles:
            if alt_colors:
                col = (255,0,0)
            else:
                col = (0,255,0)
        else:
            if alt_colors:
                return img
            col = (0,0,255)
        img = cv2.polylines(img, pts.astype(int), True, col, thickness=2)
        # img = cv2.drawMarker(img, pts[0,2].astype(int), (255,255,255))

        for pt in pts.squeeze():
            midpoint += pt

        midpoint /= 4.
        img = cv2.putText(img, str(d.tag_id), midpoint.astype(int), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

    return img

def calculate_april_objpoints(valid_tiles):
    APRIL_DIM = (6,6)
    TAG_SIZE = 0.055
    TAG_SPACE = 0.0165

    ret = np.zeros((len(valid_tiles), 4, 3), dtype=float)
    for i, tile_id in enumerate(valid_tiles):

        col = tile_id // APRIL_DIM[0]
        row = tile_id % APRIL_DIM[0]

        # Top left position
        tl = [TAG_SPACE * (col+1) + (TAG_SIZE * col),     TAG_SPACE * (row+1) + (TAG_SIZE * row    ), 0.]
        bl = [TAG_SPACE * (col+1) + (TAG_SIZE * col),     TAG_SPACE * (row+1) + (TAG_SIZE * (row+1)), 0.]
        br = [TAG_SPACE * (col+1) + (TAG_SIZE * (col+1)), TAG_SPACE * (row+1) + (TAG_SIZE * (row+1)), 0.]
        tr = [TAG_SPACE * (col+1) + (TAG_SIZE * (col+1)), TAG_SPACE * (row+1) + (TAG_SIZE * row    ), 0.]


        ret[i,:,:] = [tl, bl, br, tr]

    return ret




def get_image(cam_id, img_id, w, h, it=None):
    img = cv2.imread(glob.glob(f"./snapshots/{cam_id}_{3 if cam_id == 2 else 0}_{str(img_id).zfill(3)}_*.png")[0])

    if it is not None: # intrinsics available, undistort
        K_new = get_new_K(140, w, h)

        if cam_id == 2:
            maps = cv2.omnidir.initUndistortRectifyMap(it.K, it.D, it.xi, np.eye(3), K_new,
                                                       (w, h), cv2.CV_32FC1, cv2.omnidir.RECTIFY_PERSPECTIVE)
        else:
            maps = cv2.fisheye.initUndistortRectifyMap(it.K, it.D, np.eye(3), K_new,
                                                       (w, h), cv2.CV_32FC1)
        img = cv2.remap(img, *maps, cv2.INTER_LINEAR)

    return img


it = (load_intrinsics(cam_id=0), load_intrinsics(cam_id=1), load_intrinsics(cam_id=2))

obj_pts = [ None for _ in range(N_img) ]
cam_pts = [ [ [] for _ in cams ] for _ in range(N_img) ]
imgs = None

for img_id in tqdm(range(N_img), desc="Calculating apriltag correspondences"):

    imgs = [ get_image(cam_id, img_id, 848, 800, it[cam_id]) for cam_id in cams ]

    dets = [ detect_april(img) for img in imgs ]

    valid_tiles = []
    for tile in range(36):
        ps = [ list(filter(lambda x: x.tag_id == tile, det)) for det in dets ]
        if all([ len(p) == 1 for p in ps ]):
            valid_tiles.append(tile)
            [ cam_pt.append(p[0].corners.squeeze()) for cam_pt, p in zip(cam_pts[img_id], ps) ]

    cam_pts[img_id] = [ np.array(cam_pt) for cam_pt in cam_pts[img_id] ]

    obj_pts[img_id] = calculate_april_objpoints(valid_tiles)
            
    imgs = [ visualize_april(img, det, valid_tiles) for img, det in zip(imgs, dets) ]
    imgs[0] = visualize_april(imgs[0], dets[1], valid_tiles, alt_colors=True)

    if preview:
        [ cv2.imshow(f"cam{id}", img) for id, img in zip(cams, imgs) ]

        key = cv2.waitKey(0)
        if key == ord('q'):
            exit()

# objpoints = np.array([ e.reshape(-1,2) for e in obj_pts ], dtype=object)
objpoints = np.array([np.array(e, dtype=np.float32).reshape(-1,3) for e in obj_pts ], dtype=object)
imgpoints0 = np.array([ np.array(e[0], dtype=np.float32).reshape(-1,2) for e in cam_pts ], dtype=object)
imgpoints1 = np.array([ np.array(e[1], dtype=np.float32).reshape(-1,2) for e in cam_pts ], dtype=object)

K_new = get_new_K(140, 848, 800)

retval, K1, D1, K2, D2, R, T, E, F = cv2.stereoCalibrate(
        objpoints, imgpoints0, imgpoints1, K_new, np.zeros(4), K_new, np.zeros(4), (848,800), 
                    flags=cv2.CALIB_FIX_INTRINSIC)

print("retval:", retval)
print("R:", R)
print("t:", T.squeeze())

# cam0_pts = np.array(cam_pts[0]).reshape(-1,2)
# cam1_pts = np.array(cam_pts[1]).reshape(-1,2)
#
#
# K_new = get_new_K(140, 848, 800)
# F, mask_f = cv2.findFundamentalMat(cam0_pts, cam1_pts, cv2.FM_LMEDS)
# E, mask_e = cv2.findEssentialMat(cam0_pts, cam1_pts, K_new, cv2.FM_LMEDS)
#
# R1, R2, t = cv2.decomposeEssentialMat(E)
#
# # print(R1)
# # print(R2)
# # print(t)
#
#
# def drawlines(img1, img2, lines, pts1, pts2):
#     r,c,_ = img1.shape
#     for r, pt1, pt2 in zip(lines, pts1, pts2):
#         color = tuple(np.random.randint(0,255,3).tolist())
#         x0,y0 = map(int, [0, -r[2]/r[1] ])
#         x1,y1 = map(int, [c, -(r[2]+r[0]*c)/r[1] ])
#         img1 = cv2.line(img1, (x0,y0), (x1,y1), color, 1)
#         img1 = cv2.circle(img1, tuple(pt1.astype(int)), 5, color, -1)
#         img2 = cv2.circle(img2, tuple(pt2.astype(int)), 5, color, -1)
#
#     return img1, img2
#
# lines1 = cv2.computeCorrespondEpilines(cam1_pts.reshape(-1,1,2), 2, F)
# lines1 = lines1.reshape(-1,3)
# img5,img6 = drawlines(imgs[0],imgs[1],lines1[::4],cam0_pts[::4],cam1_pts[::4])
#
# cv2.imshow("Preview1", img5)
# cv2.imshow("Preview2", img6)
# cv2.waitKey(0)
#
#
