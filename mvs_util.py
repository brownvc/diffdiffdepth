import numpy as np
import torch
from scipy.interpolate import RectBivariateSpline
from PIL import Image
import os
from skimage.color import rgb2gray
from skimage import filters
import matplotlib.pylab as plt

def load_pts3d(fin):
    xyz = []
    rgb = []
    idx = []
    
    with open(fin, 'r') as freader:
        for line in freader:
            tokens = line.split()

            if not tokens or tokens[0] == '#':
                continue
            else:
                idx.append(int(tokens[0]))
                xyz.append([float(i) for i in tokens[1:4]])
                rgb.append([float(i) for i in tokens[4:7]])

    idx = np.array(idx)
    xyz = np.array(xyz)
    rgb = np.array(rgb)

    maxIdx = np.max(idx) + 1

    rgb_ = np.ones( (maxIdx, 3) ) * -1
    xyz_ = np.ones( (maxIdx, 3) ) * -1
    idx_ = np.ones( (maxIdx, ) ) * -1
    
    rgb_[idx, :] = rgb
    xyz_[idx, :] = xyz
    idx_[idx] = idx
    return (xyz_, rgb_, idx_)

def load_poses(fin, imfolder=None):
    poses = []
    invposes = []
    
    qwxyz = []
    txyz = []
    imgs = []
    img_names = []
    
    with open(fin, 'r') as freader:
        skipLine = False

        for line in freader:
            tokens = line.split()

            if not tokens or tokens[0] == '#':
                continue
            elif skipLine:
                skipLine = False
                continue
            else:
                qwxyz = np.array([float(i) for i in tokens[1:5]]) # rotation as a quaternion
                txyz  = np.array([float(i) for i in tokens[5:8]]) # translation

                imfile = tokens[-1]
                if imfolder:
                    imgs.append(np.asarray(Image.open(os.path.join(imfolder, imfile))))
                img_names.append(imfile)  
                pose = np.identity(4)
                pose[0:3, 0:3] = quaternion2mat(qwxyz)
                pose[0:3, -1]  = txyz
                poses.append(pose)

                invpose = np.linalg.inv(pose)
                invposes.append(invpose)
                
                skipLine = True

    return (np.array(poses), np.array(invposes), img_names, np.array(imgs))


def load_cameras(fin):
    K = []

    with open(fin, 'r') as freader:

        for line in freader:
            tokens = line.split()
            if not tokens or tokens[0] == '#':
                continue
            else:
                # FULL_OPENCV camera model: resx, resy, fx, fy, cx, cy, k1, k2, p1, p2, k3, k4, k5, k6
                cam = np.array([float(i) for i in tokens[2:]])
                K.append(cam)

    return np.array(K)

def load_images(fin):
    img_files = sorted([f for f in os.listdir(fin) if os.path.isfile(os.path.join(fin, f))])
    img = [np.array(Image.open(fin + f)) for f in img_files]
    return np.array(img)

def visibility4view(fin, view):
    V = []
    with open(fin, 'r') as freader:
        skipLine = True
        viewIdx = 0
        
        for line in freader:
            tokens = line.split()

            if not tokens or tokens[0] == '#':
                continue
            elif skipLine:
                skipLine = False
                continue
            elif viewIdx != view :
                viewIdx += 1
                skipLine = True
                continue
            else:
                vidx = np.array([i for i in range(2, len(tokens), 3)])
                V = np.array([float(i) for i in tokens])[vidx].astype(int)
                break

    V = V[V != -1]
    return np.array(V)

def visibility_from_color(pts2D, rgb, im, t):
    # Sample image at pts2D
    h, w, _ = im.shape
    x = np.arange(0, w)
    y = np.arange(0, h)

    f_r = RectBivariateSpline(y, x, im[:, :, 0])
    f_g = RectBivariateSpline(y, x, im[:, :, 1])
    f_b = RectBivariateSpline(y, x, im[:, :, 2])

    rgb_s = np.stack((f_r( pts2D[:, 1], pts2D[:, 0], grid=False), \
                      f_g( pts2D[:, 1], pts2D[:, 0], grid=False), \
                      f_b( pts2D[:, 1], pts2D[:, 0], grid=False)), axis=1)

    # Calculate difference of sampled values and input rgb
    diff = np.sqrt(np.sum((rgb - rgb_s) ** 2, axis=1))

    # Identify occluded points based on rgb difference threshold t
    idx = diff < t
    return idx

def world2view(pts3d, pose, cam, pinhole=1):
    # Camera space points
    xc = pts3d[:, 0] * pose[0, 0] + pts3d[:, 1] * pose[0, 1] + pts3d[:, 2] * pose[0, 2] + pose[0, 3]
    yc = pts3d[:, 0] * pose[1, 0] + pts3d[:, 1] * pose[1, 1] + pts3d[:, 2] * pose[1, 2] + pose[1, 3]
    zc = pts3d[:, 0] * pose[2, 0] + pts3d[:, 1] * pose[2, 1] + pts3d[:, 2] * pose[2, 2] + pose[2, 3]

    # Image space points
    x0 = xc / zc
    y0 = yc / zc
    r = x0 ** 2 + y0 ** 2

    # Ignoring radial and tangential distortion
    x1 = x0 
    y1 = y0 

    if pinhole:
        u = cam[2] * x1 + cam[3]
        v = cam[2] * y1 + cam[4]

    else:
        u = cam[2] * x1 + cam[4]
        v = cam[3] * y1 + cam[5]

    return np.stack((u, v, zc), axis=1)


def world2views(x, y, z, poses, cam, pinhole=1):
    x = x.unsqueeze(1).expand(-1, poses.shape[0], -1, -1)
    y = y.unsqueeze(1).expand(-1, poses.shape[0], -1, -1)
    z = z.unsqueeze(1).expand(-1, poses.shape[0], -1, -1)

    poses = poses.unsqueeze(0).unsqueeze(-1).unsqueeze(-1) # add channels and spatial dimensions

    # Camera space points
    xc = x * poses[:, :, 0, 0] + y * poses[:, :, 0, 1] + z * poses[:, :, 0, 2] + poses[:, :, 0, 3]
    yc = x * poses[:, :, 1, 0] + y * poses[:, :, 1, 1] + z * poses[:, :, 1, 2] + poses[:, :, 1, 3]
    zc = x * poses[:, :, 2, 0] + y * poses[:, :, 2, 1] + z * poses[:, :, 2, 2] + poses[:, :, 2, 3]
        
    # Image space points
    x0 = xc / zc
    y0 = yc / zc
    r = x0 ** 2 + y0 ** 2

    # Ignoring radial and tangential distortion for now to make the inverse operation simpler
    if pinhole:
        u = cam[2] * x0 + cam[3]
        v = cam[2] * y0 + cam[4]

    else:
        u = cam[2] * x0 + cam[4]
        v = cam[3] * y0 + cam[5]

    return (u, v)

def view2world(u, v, d, invpose, cam, pinhole=1):

    # Image space points
    if pinhole:
        x0 = (u - cam[3]) / cam[2]
        y0 = (v - cam[4]) / cam[2]
    else:
        x0 = (u - cam[4]) / cam[2]
        y0 = (v - cam[5]) / cam[3]
    
    # Camera space
    zc = d
    xc = x0 * zc
    yc = y0 * zc

    x = xc * invpose[0, 0] + yc * invpose[0, 1] + zc * invpose[0, 2] + invpose[0, 3]
    y = xc * invpose[1, 0] + yc * invpose[1, 1] + zc * invpose[1, 2] + invpose[1, 3]
    z = xc * invpose[2, 0] + yc * invpose[2, 1] + zc * invpose[2, 2] + invpose[2, 3]

    return (x, y, z)

def quaternion2mat(q):
    rot = np.zeros((3, 3))
    rot[0, 0] = 1 - 2 * q[2] ** 2 - 2 * q[3] ** 2
    rot[0, 1] = 2 * q[1] * q[2] - 2 * q[3] * q[0]
    rot[0, 2] = 2 * q[1] * q[3] + 2 * q[2] * q[0]
    rot[1, 0] = 2 * q[1] * q[2] + 2 * q[3] * q[0]
    rot[1, 1] = 1 - 2 * q[1] ** 2 - 2 * q[3] ** 2
    rot[1, 2] = 2 * q[2] * q[3] - 2 * q[1] * q[0]
    rot[2, 0] = 2 * q[1] * q[3] - 2 * q[2] * q[0]
    rot[2, 1] = 2 * q[2] * q[3] + 2 * q[1] * q[0]
    rot[2, 2] = 1 - 2 * q[1] ** 2 - 2 * q[2] ** 2
    return rot

def densify_grid(pts2d, z, sz):
    stride = 10
    x, y = np.meshgrid( np.arange(0, sz[1], stride), np.arange(0, sz[0], stride) )

    x = x.reshape(1, np.int(sz[0] * sz[1] / (stride * stride)))
    dx = np.expand_dims(pts2d[:, 0], 1) - x

    y = y.reshape(1, np.int(sz[0] * sz[1] / (stride * stride)))
    dy = np.expand_dims(pts2d[:, 1], 1) - y

    d = dx ** 2 + dy ** 2
    midx = np.argmin(d, axis=0)

    return np.transpose(np.stack((x.flatten(), y.flatten(), z[midx]), axis=0))

def densify_edge(pts2D, d, dmap, I, densify_factor=2):
    edges = feature.canny(I[0, :, :].cpu().detach().numpy().squeeze())
    
    h, w = dmap.shape
    num_new_pts = pts2D.shape[0] * densify_factor - pts2D.shape[0]
    new_pts_y, new_pts_x = np.nonzero(edges)
    o = np.random.permutation( new_pts_x.shape[0] )
    new_pts_x = torch.tensor(new_pts_x[o][:num_new_pts])
    new_pts_y = torch.tensor(new_pts_y[o][:num_new_pts])
    new_pts_z = dmap[ new_pts_y, new_pts_x ]

    P = torch.cat( (pts2D, torch.cat( (new_pts_x.float().unsqueeze(1), new_pts_y.float().unsqueeze(1)), dim=1)), dim=0)
    D = torch.cat( (d, new_pts_z), dim=0)
    return P, D

def densify_rand(pts2D, d, dmap, densify_factor=2):
    h, w = dmap.shape
    num_new_pts = pts2D.shape[0] * densify_factor - pts2D.shape[0]
    new_pts_x = (torch.rand( (num_new_pts, ) ) * w).long()
    new_pts_y = (torch.rand( (num_new_pts, ) ) * h).long()
    new_pts_z = dmap[ new_pts_y, new_pts_x ]

    P = torch.cat( (pts2D, torch.cat( (new_pts_x.float().unsqueeze(1), new_pts_y.float().unsqueeze(1)), dim=1)), dim=0)
    D = torch.cat( (d, new_pts_z), dim=0)
    return P, D
