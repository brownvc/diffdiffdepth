import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import argparse
import numpy as np
import mvs_util
import config
from skimage import feature
from skimage.color import rgb2gray
from skimage import filters
from loss import Loss
import os

def optimize(base_dir, src_img, out_dir):
    pts3D, rgb, idx = mvs_util.load_pts3d(os.path.join(base_dir, 'sparse/0/points3D.txt'))
    poses, invposes, I_names, I = mvs_util.load_poses(os.path.join(base_dir, 'sparse/0/images.txt'), os.path.join(base_dir, 'images'))
    cam = mvs_util.load_cameras(os.path.join(base_dir, 'sparse/0/cameras.txt'))
    
    _, h, w, _ = I.shape

    try: # TODO: test
        cposeidx = I_names.index(src_img)
    except ValueError:
        print("ERROR! File %s does not exist in directory %d. Please make sure you entered the name correctly." % (src_img, os.path.abspath(base_dir)))
        return
        
    pts2D = mvs_util.world2view(pts3D, poses[cposeidx, :, :], cam.flatten()) # 2D Projection of 3D points

    # Select only points visible in src_img
    V = mvs_util.visibility4view(os.path.join(base_dir, 'sparse/0/images.txt'), cposeidx) 
    pts3D = pts3D[V, :]
    pts2D = pts2D[V, :]

    poses = torch.tensor(poses[np.r_[:cposeidx, cposeidx+1:poses.shape[0]], :, :], dtype=torch.float32, device=config.device) # poses of all images except src_img
    I_invpose = torch.tensor(invposes[cposeidx, :, :], dtype=torch.float32, device=config.device) # inverse pose of src_img
    cam = torch.tensor(cam, dtype=torch.float32, device=config.device)

    # The data loading code assumes a single batch for now.
    batch_sz = 1
    P = torch.tensor(pts2D[:, :-1], dtype=torch.float32, device=config.device).unsqueeze(0).repeat(batch_sz, 1, 1) # label xy positions
    D = torch.tensor(pts2D[:, -1], dtype=torch.float32, device=config.device).unsqueeze(0).repeat(batch_sz, 1)  # label values
    W = torch.zeros( D.shape ) # label data weights

    # The smoothness weight at each pixel is initialized using the scaled gradient magnitude
    # (To avoid negative weights, the actual weight is the negative exponent of the gradient. See below...)
    S = torch.tensor( 6 * filters.sobel( rgb2gray(I[cposeidx, :, :, :]) ), dtype=torch.float32, device=config.device).unsqueeze(0).repeat(batch_sz, 1, 1)
    S[:, [0, -1], :] = torch.max(S)
    S[:, :, [0, -1]] = torch.max(S)

    I_center = torch.tensor( I[cposeidx, :, :, :], dtype=torch.float32, device=config.device).unsqueeze(0).repeat(batch_sz, 1, 1, 1).permute(0, 3, 1, 2) / 255 # src_img
    # multiview image set:
    I = torch.tensor( I[np.r_[:cposeidx, cposeidx+1:I.shape[0]], :, :, :], dtype=torch.float32, device=config.device).unsqueeze(0).repeat(batch_sz, 1, 1, 1, 1).permute(0, 1, 4, 2, 3) / 255

    step = 1
    max_steps = (config.optim_xy + config.optim_disp + config.optim_dw + config.optim_smoothness) * config.num_iter * config.num_passes
    li = []
    
    def optim_param(optimizer):
        nonlocal step
        for iter in range(config.num_iter):
            loss = Loss()

            # Select a random set of views to project to
            views = torch.randperm( I.shape[1] ).long()[:config.nviews_reproj]
            I_rand = I[:, views, :, :, :]
            poses_rand = poses[views, :, :]

            optimizer.zero_grad()

            sx = torch.exp(-S)
            sy = torch.exp(-S)

            l, o = loss.loss_mvs(P, D, torch.exp(-W), sx, sy, I_center, I_rand, poses_rand, I_invpose, cam,
                                 torch.tensor(config.tile_max_pts), config.factor, config.loss_VGG, config.loss_grad, config.loss_smoothness)
            l.backward()
            optimizer_s.step()
            li.append(l.item())
            
            print(("%d/%d; Loss: %f" % (step, max_steps, l.item()) ))
            step = step + 1

            if config.log_progress or step >= max_steps:
                if not os.path.exists(out_dir):
                    os.makedirs(out_dir)
                np.save( os.path.join(out_dir, 'o_%d.npy' % step), o.cpu().detach().numpy().squeeze())

            if step >= max_steps:
                if not os.path.exists(out_dir):
                    os.makedirs(out_dir)
                np.save( os.path.join(out_dir, 'loss.npy'), np.asarray(li) )
                np.save( os.path.join(out_dir, '%.output.npy' % src_img), o.cpu().detach().numpy().squeeze())
            
    for npass in range( config.num_passes ):

        #
        # Optimize Gradients
        if config.optim_smoothness:
            torch.cuda.empty_cache()        
            S.requires_grad = True
            optimizer_s = optim.Adam([S], lr=config.learning_rate_smoothness * (0.9 ** npass) ) 
            optim_param(optimizer_s)
            
            # COLMAP points are usually very sparse.
            # We densify the points by sampling along edges after the very first gradient pass.
            if npass == 0:
                P, D = mvs_util.densify_edge(P.squeeze(0), D.squeeze(0), o.squeeze(0), I_center.squeeze(0), 4)
                P = P.unsqueeze(0).clone().detach()
                D = D.unsqueeze(0).clone().detach()
                W = torch.zeros( D.shape )

            S.requires_grad=False

        #
        # Optimize Label Value (Depth/Disparity)
        if config.optim_disp:
            torch.cuda.empty_cache()        
            D.requires_grad = True
            optimizer_d = optim.Adam([D], lr=config.learning_rate_disp * (0.9 ** npass))
            optim_param(optimizer_d)
            D.requires_grad = False

        #
        # Optimize Data Weights
        if config.optim_dweights:
            torch.cuda.empty_cache()        
            W.requires_grad = True
            optimizer_w = optim.Adam([W], lr=learning_rate_dw * (0.8 ** npass))
            optim_param(optimizer_w)
            W.requires_grad = False

        #
        # Optimize Point Positions
        if optim_xy:
            torch.cuda.empty_cache()   
            P.requires_grad = True
            optimizer_xy = optim.Adam([P], lr=learning_rate_xy * (0.75 ** npass))
            optim_param(optimizer_xy)
            P.requires_grad = False


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Optimize diffusion parameters for multi-view stereo')
    parser.add_argument('--input_dir', help='Base COLMAP project directory with poses and camera parameters', required=True)
    parser.add_argument('--src_img', help='Name of the image for which to compute depth', required=True)
    parser.add_argument('--output_dir', help='Name of output directory for saving results', required=True)
    parser.add_argument('--optim_xy', type=bool, default=config.optim_xy, help='Optimize label positions')
    parser.add_argument('--optim_disp', type=bool, default=config.optim_disp, help='Optimize label value')
    parser.add_argument('--optim_smoothness', type=bool, default=config.optim_smoothness, help='Optimize smoothness weights')
    parser.add_argument('--optim_dw', type=bool, default=config.optim_dw, help='Optimize label data weights')
    parser.add_argument('--nviews_reproj', type=int, default=config.nviews_reproj, help='Maximum number of views to use for reprojection loss')
    parser.add_argument('--factor', type=int, default=config.factor, help='Scale factor')
    parser.add_argument('--num_passes', type=int, default=config.num_passes, help='Number of optimization passes')
    parser.add_argument('--num_iter', type=int, default=config.num_iter, help='Number of iterations in each optimization pass')
    parser.add_argument('--log_progress', type=bool, default=config.log_progress, help='Log intermediate output')
    args = parser.parse_args()

    config.num_passes = args.num_passes
    config.num_iter = args.num_iter
    config.optim_xy = args.optim_xy
    config.optim_disp = args.optim_disp
    config.optim_dw = args.optim_dw
    config.optim_smoothness = args.optim_smoothness
    config.nview_reproj = args.nviews_reproj
    config.factor = args.factor
    config.log_progress = args.log_progress
    
    if torch.cuda.is_available():
        config.device = torch.device("cuda:0")
        torch.cuda.empty_cache()
        torch.set_default_tensor_type(torch.cuda.FloatTensor)
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True
        print("Running on the GPU")
    else:
        config.device = torch.device("cpu")
        print("Running on the CPU")

    optimize(args.input_dir, args.src_img, args.output_dir)
