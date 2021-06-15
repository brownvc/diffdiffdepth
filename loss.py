import torch
import torch.nn as nn
import torch.nn.functional as F
from LAHBPCG import PCG
from points2raster import points2raster
import pytorch_ssim
import vggfeatures
from mvs_util import view2world, world2views
import numpy as np
import config

class Loss():
    def __init__(self):
        self.niter = 5
        self.emin = 0.01
        self.wg = 1.0e-2 #1.0e-1
        self.l1 = torch.nn.L1Loss()

    # Edge aware smoothness loss implementation is adapted from: https://github.com/anuragranj/cc
    def edge_aware_smoothness_per_pixel(self, img, pred):
        """ A measure of how closely the gradients of a predicted disparity/depth map match the 
        gradients of the RGB image. 

        Args:
          img (c x 3 x h x w tensor): RGB image
          pred (c x h x w tensor): predicted depth/disparity

        Returns:
          c x 1 tensor: measure of gradient matching (smoothness loss)
        """
        
        def gradient_y(img):
            gy = torch.cat( [F.conv2d(img[:, i, :, :].unsqueeze(0), torch.Tensor([[1, 2, 1], [0, 0, 0], [-1, -2, -1]]).view((1, 1, 3, 3)), padding=1) for i in range(img.shape[1])], 1)
            return gy

        def gradient_x(img):
            gx = torch.cat( [F.conv2d(img[:, i, :, :].unsqueeze(0), torch.Tensor([[1, 0, -1], [2, 0, -2], [1, 0, -1]]).view((1, 1, 3, 3)), padding=1) for i in range(img.shape[1])], 1)
            return gx
        
        pred_gradients_x = gradient_x(pred)
        pred_gradients_y = gradient_y(pred)

        image_gradients_x = gradient_x(img)
        image_gradients_y = gradient_y(img)

        weights_x = torch.exp(-torch.mean(torch.abs(image_gradients_x), 1, keepdim=True))
        weights_y = torch.exp(-torch.mean(torch.abs(image_gradients_y), 1, keepdim=True))
                
        smoothness_x = torch.abs(pred_gradients_x) * weights_x
        smoothness_y = torch.abs(pred_gradients_y) * weights_y

        return torch.mean(smoothness_x) + torch.mean(smoothness_y)

    
    def loss_mvs(self, pts_in, disp, dw, gx, gy, imc, im, imposes, invposec, cam, tileMaxPts,
                 scaleFactor=1.0,
                 use_vgg_features=False,
                 gradient_loss=False,
                 smoothness_loss=False):
        """ Loss function for depth computation from multi-view stereo images

        Args:
           pts_in (c x n x 2 tensor): 2D position of labels in image space
           disp( c x n x 1 tensor): Depth labels
           dw (c x n x 1 tensor): Data weight (confidence)
           gx (c x h x w tensor): Smoothness weight, x
           gy (c x h x w tensor): Smoothness weight, y
           imc (c x 3 x h x w tensor): Central image for which depth is optimized
           im (c x m x 3 x h x w tensor): Set of multi-view images used for reprojection 
           imposes (c x m x 3 x 4 tensor): Poses of multi-view images
           invposec(c x 3 x 4 tensor): Inverse of the pose of the central view
           cam (c x ? tensor): Intrinsic camera parameters
           tileMaxPts (int): Maximum number of points per tile (see points2raster.py)
           scaleFactor (float): Scaling factor for input image
           use_vgg_features (bool): Use VGG rather than RGB features in reprojection loss
           gradient_loss (bool): Use gradient loss term
           smoothness_loss (bool): Use smoothness loss term

        Returns:
           float: Loss for current set of parameters.
        """
        
        c, nproj, _, h, w = im.shape

        pts = pts_in * scaleFactor
        scale = nn.Upsample(scale_factor = scaleFactor, mode='bilinear', align_corners=True)
        gx = scale(gx.unsqueeze(0)).squeeze(0)
        gy = scale(gy.unsqueeze(0)).squeeze(0)

        d, k = points2raster(pts, disp, dw, (h * scaleFactor, w * scaleFactor), tileMaxPts, scaleFactor, 5)
        self.solver = PCG(c, h * scaleFactor, w * scaleFactor)
        self.solver.set_constraints(d, # data term
                                    torch.zeros((c, h * scaleFactor, w * scaleFactor), dtype=torch.float32), # target gradients, x
                                    torch.zeros((c, h * scaleFactor, w * scaleFactor), dtype=torch.float32), # target gradients, y
                                    k, # data weight
                                    gx, # gradient weights, x
                                    gy) # gradient weights, y
        o = torch.zeros((c, h * scaleFactor, w * scaleFactor), requires_grad=True)
        o = self.solver.solve(o, self.niter, self.emin).float()
        o = F.interpolate(o.unsqueeze(0), [h, w], mode='bilinear', align_corners=True).squeeze(0)

        #
        # Identify and exclude occluded pixels from reprojection loss computation
        # This is done by forward projecting and identifying the rounded pixel positions
        # that receive more than one depth label. The labels with large depth are occluded.
        y, x = torch.meshgrid(torch.arange(0, h), torch.arange(0, w))
        od = o.clone()
        xw, yw, zw = view2world(x, y, od, invposec, cam.flatten())
        ox, oy = world2views(xw, yw, zw, imposes, cam.flatten())
        od = od.unsqueeze(1).expand(-1, nproj, -1, -1)
        
        rx = torch.round(ox)
        ry = torch.round(oy)
        idx = torch.logical_or(torch.logical_or(ox < 0, ox >  w - 1), torch.logical_or(oy < 0, oy > h - 1))
        rx[idx] = 0
        ry[idx] = 0

        # Discretize depth 
        nDepths = 32
        od = ((od - torch.min(od)) / (torch.max(od) - torch.min(od)) * (nDepths - 1)).int()
        comp = torch.ones(c, nproj, h * w) * (1e10)
        mask = torch.zeros(c, nproj, h, w)
        mask[idx] = 1
        
        # Estimate occlusion masks
        for i in range(nDepths): 
            cIdx = od != i
            xi, yi = rx.clone().detach(), ry.clone().detach()
            xi[cIdx] = 0
            yi[cIdx] = 0

            occIdx = torch.gather(comp, -1, torch.reshape((yi * w + xi).long(), (c, nproj, h * w))) < i
            occIdx = occIdx.reshape(c, nproj, h, w)

            mask[torch.logical_and(torch.logical_not(cIdx), occIdx)] = 1

            xi[occIdx] = 0
            yi[occIdx] = 0

            comp.scatter_(-1, torch.reshape((yi * w + xi).long(), (c, nproj, h * w)), i)

        #
        # Reprojection Loss
        # Computed by backward projection of non-center views onto center view using predeicted depth
        ox = ox / float(w - 1) * 2 - 1.0 # Normalize grid coordinates to [-1, 1]
        oy = oy / float(h - 1) * 2 - 1.0

        u = torch.sum(1 - mask, 1)
        u[u == 0] = 1

        rpj = torch.zeros(im.shape)

        for i in range(nproj):
            grid = torch.stack((ox[:, i, :, :], oy[:, i, :, :]), -1)
            rpj[:, i, :, :, :] = F.grid_sample(im[:, i, :, :, :], grid, align_corners=True)

        imcp = imc.unsqueeze(1).expand(-1, nproj, -1, -1, -1)
        mask = mask.unsqueeze(2).expand(-1, -1, 3, -1, -1).bool()

        rpj[mask] = 0
        a = torch.mean(rpj, 1).squeeze(0).cpu().detach().numpy().squeeze()
        a = np.uint8(a * 255)
        a = np.uint8(np.transpose(a, (1, 2, 0)))

        # The occluded pixels have zero loss
        rpj[mask] = imcp[mask]

        if use_vgg_features:
            vgg16_4 = vggfeatures.VggFeatures()
            vgg16_4.to(device=config.device)
            loss_reproj = torch.abs(vgg16_4(rpj[0, :, :, :, :]) - vgg16_4(imcp[0, :, :, :, :]))
        else:
            loss_reproj = torch.abs(rpj[0, :, :, :, :] - imcp[0, :, :, :, :])

        loss_reproj = torch.mean(loss_reproj, 1)

        # Structural Self-similarity loss
        ssim = pytorch_ssim.SSIM()
        loss_ssim = 0.05 * ssim(rpj.squeeze(0), imcp.squeeze(0))

        # Gradient loss enforced strong gradients in the error map
        if gradient_loss:
            gx_o = F.conv2d(loss_reproj.unsqueeze(0), torch.Tensor([[1, 0, -1], [2, 0, -2], [1, 0, -1]]).view((1, 1, 3, 3)).expand(-1, nproj, -1, -1), padding=1)
            gy_o = F.conv2d(loss_reproj.unsqueeze(0), torch.Tensor([[1, 2, 1], [0, 0, 0], [-1, -2, -1]]).view((1, 1, 3, 3)).expand(-1, nproj, -1, -1), padding=1)
            loss_g = -torch.mean(torch.sqrt(torch.pow(gx_o, 2) + torch.pow(gy_o, 2) + 1e-10))
        else:
            loss_g = 0

        loss_reproj = torch.mean(torch.sum(loss_reproj, 0) / (u + 1e-10))

        # Smoothness loss enforces the gradients of the depth map to match the gradients of the RGB image
        if smoothness_loss:
            Ls = self.edge_aware_smoothness_per_pixel(imc, o.unsqueeze(0))
            Ls = Ls * 0.0075 
        else:
            Ls = 0.0

        return(loss_reproj + self.wg * loss_g + Ls + loss_ssim, o)

    
    def loss_lf(self, pts_in, disp, dw, gx, gy, imc, im, imdxy, tileMaxPts,
                scaleFactor=1.0,
                use_vgg_features=False,
                gradient_loss=False,
                smoothness_loss=False):
        """ Loss function for disparity computation from light fields

        Args:
           pts_in (c x n x 2 tensor): 2D position of labels in image space
           disp( c x n x 1 tensor): Depth labels
           dw (c x n x 1 tensor): Data weight (confidence)
           gx (c x h x w tensor): Smoothness weight, x
           gy (c x h x w tensor): Smoothness weight, y
           imc (c x 3 x h x w tensor): Central image for which depth is optimized
           im (c x m x 3 x h x w tensor): Set of light field images used for reprojection 
           imdxy (c x m x 2 tensor): The uv offset of each light field image
           tileMaxPts (int): Maximum number of points per tile (see points2raster.py)
           scaleFactor (float): Scaling factor for input image
           use_vgg_features (bool): Use VGG rather than RGB features in reprojection loss
           gradient_loss (bool): Use gradient loss term
           smoothness_loss (bool): Use smoothness loss term

        Returns:
           float: Loss for current set of parameters.
        """

        c, nproj, _, h, w = im.shape

        pts = pts_in * scaleFactor
        scale = nn.Upsample(scale_factor = scaleFactor, mode='bilinear', align_corners=True)
        gx = scale(gx.unsqueeze(0)).squeeze(0)
        gy = scale(gy.unsqueeze(0)).squeeze(0)

        d, k = points2raster(pts, disp, dw, (h * scaleFactor, w * scaleFactor), tileMaxPts, scaleFactor, 3)
        
        self.solver = PCG(c, h * scaleFactor, w * scaleFactor)
        self.solver.set_constraints(d, # data term
                                    torch.zeros((c, h * scaleFactor, w * scaleFactor), dtype=torch.float32), # target gradients, x
                                    torch.zeros((c, h * scaleFactor, w * scaleFactor), dtype=torch.float32), # target gradients, y
                                    k, # data weight
                                    gx, # gradient weights, x
                                    gy) # gradient weights, y
        o = torch.zeros((c, h * scaleFactor, w * scaleFactor), requires_grad=True)
        o = self.solver.solve(o, self.niter, self.emin).float()
        o = F.interpolate(o.unsqueeze(0), [h, w], mode='bilinear', align_corners=True).squeeze(0)

        #
        # Identify and exclude occluded pixels from reprojection loss computation
        # This is done by forward projecting and identifying the rounded pixel positions
        # that receive more than one depth label. The labels with large depth are occluded.
        y, x = torch.meshgrid(torch.arange(0, h), torch.arange(0, w))
        od = o.clone().unsqueeze(1).expand(-1, nproj, -1, -1)

        ox = od * imdxy[:, :, 0].unsqueeze(-1).unsqueeze(-1) + x
        oy = od * imdxy[:, :, 1].unsqueeze(-1).unsqueeze(-1) + y

        rx = torch.round(ox)
        ry = torch.round(oy)
        idx = torch.logical_or(torch.logical_or(ox < 0, ox >  w - 1), torch.logical_or(oy < 0, oy > h - 1))
        rx[idx] = 0
        ry[idx] = 0

        # Discretize od
        nDepths = 32  
        maxAbsDisparity = 5 
        od[ od < -maxAbsDisparity ] = -maxAbsDisparity
        od[ od > maxAbsDisparity ] = maxAbsDisparity
        od = ((od - torch.min(od)) / (torch.max(od) - torch.min(od)) * (nDepths - 1)).int()

        comp = torch.ones(c, nproj, h * w) * (-1)
        mask = torch.zeros(c, nproj, h, w)
        mask[idx] = 1
        
        # Estimate occlusion masks
        for i in range(nDepths - 1, -1, -1):
            cIdx = od != i
            xi, yi = rx.clone().detach(), ry.clone().detach()
            xi[cIdx] = 0
            yi[cIdx] = 0
            
            occIdx = torch.gather(comp, -1, torch.reshape((yi * w + xi).long(), (c, nproj, h * w))) > i
            occIdx = occIdx.reshape(c, nproj, h, w)

            mask[torch.logical_and(torch.logical_not(cIdx), occIdx)] = 1
            
            xi[occIdx] = 0
            yi[occIdx] = 0
            comp.scatter_(-1, torch.reshape((yi * w + xi).long(), (c, nproj, h * w)), i)

            
        #
        # Reprojection Loss
        # Computed by backward projection of non-center views onto center view using predeicted depth
        ox = ox / float(w - 1) * 2 - 1.0 # Normalize grid coordinates to [-1, 1]
        oy = oy / float(h - 1) * 2 - 1.0

        u = torch.sum(1 - mask, 1)
        u[u == 0] = 1

        rpj = torch.zeros(im.shape)

        for i in range(nproj):
            grid = torch.stack((ox[:, i, :, :], oy[:, i, :, :]), -1)
            rpj[:, i, :, :, :] = F.grid_sample(im[:, i, :, :, :], grid, align_corners=True)

        imcp = imc.unsqueeze(1).expand(-1, nproj, -1, -1, -1)
        mask = mask.unsqueeze(2).expand(-1, -1, 3, -1, -1).bool()
        rpj[mask] = imcp[mask] # The occluded pixels have zero loss

        if use_vgg_features:
            vgg16_4 = vggfeatures.VggFeatures()
            vgg16_4.cuda()
            loss_reproj = torch.abs(vgg16_4(rpj[0, :, :, :, :]) - vgg16_4(imcp[0, :, :, :, :]))
        else:
            loss_reproj = torch.abs(rpj[0, :, :, :, :] - imcp[0, :, :, :, :])
        
        loss_reproj = torch.mean(loss_reproj, 1)

        # Gradient loss enforced strong gradients in the error map
        if gradient_loss:
            gx_o = F.conv2d(loss_reproj.unsqueeze(0), torch.Tensor([[1, 0, -1], [2, 0, -2], [1, 0, -1]]).view((1, 1, 3, 3)).expand(-1, nproj, -1, -1), padding=1)
            gy_o = F.conv2d(loss_reproj.unsqueeze(0), torch.Tensor([[1, 2, 1], [0, 0, 0], [-1, -2, -1]]).view((1, 1, 3, 3)).expand(-1, nproj, -1, -1), padding=1)
            loss_g = -torch.mean(torch.sqrt(torch.pow(gx_o, 2) + torch.pow(gy_o, 2) + 1e-10))
        else:
            loss_g = 0

        loss_reproj = torch.mean(torch.sum(loss_reproj, 0) / (u + 1e-10))

        # Smoothness loss enforces the gradients of the depth map to match the gradients of the RGB image
        if smoothness_loss:
            Ls = self.edge_aware_smoothness_per_pixel(imc, o.unsqueeze(0))
            Ls = Ls * 0.05 
        else:
            Ls = 0.0

        return(loss_reproj + self.wg * loss_g + Ls, o)

    
    def loss_supervised(self, pts_in, disp, dw, gx, gy, g, imc, gt, tileMaxPts, scaleFactor):
        """ Supervised loss using ground truth depth. Used for evaluation purposes.

        Args:
           pts_in (c x n x 2 tensor): 2D position of labels in image space
           disp( c x n x 1 tensor): Depth labels
           dw (c x n x 1 tensor): Data weight (confidence)
           gx (c x h x w tensor): Smoothness weight, x
           gy (c x h x w tensor): Smoothness weight, y
           imc (c x 3 x h x w tensor): Central image for which depth is optimized
           gt (c x h x w tensor): ground truth depth
           imdxy (c x m x 2 tensor): The uv offset of each light field image
           tileMaxPts (int): Maximum number of points per tile (see points2raster.py)
           scaleFactor (float): Scaling factor for input image

        Returns:
           float: Loss for current set of parameters.
        """

        c, h, w = gt.shape

        pts = pts_in * scaleFactor
        scale = nn.Upsample(scale_factor = scaleFactor, mode='bilinear', align_corners=True)
        gx = scale(gx.unsqueeze(0)).squeeze(0)
        gy = scale(gy.unsqueeze(0)).squeeze(0)

        d, k = points2raster(pts, disp, dw, (h * scaleFactor, w * scaleFactor), tileMaxPts, scaleFactor)
        
        self.solver = PCG(c, h * scaleFactor, w * scaleFactor)
        self.solver.set_constraints(d, # data term
                                    torch.zeros((c, h * scaleFactor, w * scaleFactor), dtype=torch.float32), # target gradients, x
                                    torch.zeros((c, h * scaleFactor, w * scaleFactor), dtype=torch.float32), # target gradients, y
                                    k, # data weight
                                    gx, # gradient weights, x
                                    gy) # gradient weights, y
        o = torch.zeros((c, h * scaleFactor, w * scaleFactor), requires_grad=True)
        o = self.solver.solve(o, self.niter, self.emin).float()
        o = F.interpolate(o.unsqueeze(0), [h, w], mode='bilinear', align_corners=True).squeeze(0)

        gx_o = F.conv2d(o.unsqueeze(0), torch.Tensor([[1, -1, 0], [0, 0, 0], [0, 0, 0]]).view((1, 1, 3, 3)), padding=1)
        gy_o = F.conv2d(o.unsqueeze(0), torch.Tensor([[1, 0, 0], [-1, 0, 0], [0, 0, 0]]).view((1, 1, 3, 3)), padding=1)
        gxy_o = torch.sqrt(torch.square(gx_o) + torch.square(gy_o) + 1e-10)
        
        gamma = 2.2
        Y = .2126 * torch.pow(imc[0, 0, :, :], gamma) + .7152 * torch.pow(imc[0, 1, :, :], gamma) + .0722 * torch.pow(imc[0, 2, :, :], gamma)
        L = 116 * torch.pow(Y, 1/3) - 16

        L = L.unsqueeze(0)
        gx_i = F.conv2d(L.unsqueeze(0), torch.Tensor([[1, -1, 0], [0, 0, 0], [0, 0, 0]]).view((1, 1, 3, 3)), padding=1)
        gy_i = F.conv2d(L.unsqueeze(0), torch.Tensor([[1, 0, 0], [-1, 0, 0], [0, 0, 0]]).view((1, 1, 3, 3)), padding=1)
        gxy_i = torch.sqrt(torch.square(gx_i) + torch.square(gy_i) + 1e-10)
        
        Ls = 0.00075 * torch.mean(torch.sum(torch.square(torch.exp(-gxy_i) * gxy_o)))

        return (self.l1(o, gt) + Ls, o)

