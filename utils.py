import numpy as np

class Intrinsics:
    w: int; h: int

    K: np.ndarray
    D: np.ndarray
    xi = None


    def __init__(self, K, D, w, h, xi=None):
        self.K = K
        self.D = D
        self.w = w; self.h = h

        if xi is not None:
            if isinstance(xi, np.ndarray):
                self.xi = xi
            else:
                self.xi = np.array([xi])

def load_intrinsics(cam_id):
    d = dict(np.load(f"cam{cam_id}_intrinsics.npz"))

    if cam_id == 2:
        meta = Intrinsics(d['K'], d['D'], int(d['w']), int(d['h']), d['xi'])
    else:
        meta = Intrinsics(d['K'], d['D'], int(d['w']), int(d['h']))

    return meta


def get_new_K(fov, w, h):
    f = np.abs(w/(2*np.tan(np.deg2rad(fov/2))))
    return np.array([[f, 0, w/2],[0, f, h/2],[0,0,1]])

def save_intrinsics(cam_id, xi=None, fx=None, fy=None, 
                    cx=None, cy=None, k1=None, k2=None, 
                    p1=None, p2=None, w=None, h=None):
    if cam_id == 2: #omnidirectional
        np.savez(f"cam{cam_id}_intrinsics.npz",
                 K=np.array([[fx, 0, cx], [0, fy, cy], [0,0,1]]),
                 D=np.array([k1,k2,p1,p2]),
                 xi=np.array([xi]), w=w, h=h)

    else: # fisheye
        np.savez(f"cam{cam_id}_intrinsics.npz",
                 K=np.array([[fx, 0, cx], [0, fy, cy], [0,0,1]]),
                 D=np.array([k1,k2,p1,p2]),
                 w=w, h=h)


