import torch
import torch.nn.functional as F

def points2raster(p, d, dw, sz, tileMaxPts, scaleFactor, winSz):
    """ Splat a sparse set of depth labels in 2D onto a discrete raster grid as 
    Gaussians with super-Gaussian weight. Handle occlusion as radiative energy 
    transfer through three-dimensional Gaussians

    Args:
      p (c x n x 2 tensor): batch of n 2D points
      d (c x n x 1 tensor): depth labels for n points
      dw (c x n x 1 tensor): data weight (confidence) of each point
      sz ( tuple ) : height, width of the 2D raster grid
      tileMaxPts (int): The function uses tiling for efficient GPU implementation.
                        Specify the max number of points per 2D tile.
      scaleFactor(int): Scaling factor for sz. 
      winSz: Size of the 2D support window used for Gaussians
    
    Returns:
      tuple: (data labels, data weights) defined on 2D raster grid
    """

    # The constants chosen here work in all cases
    PI = torch.acos(torch.zeros(1)).item() * 2
    wsz = torch.tensor(winSz, dtype=torch.int32)
    sig_d = torch.tensor(1.3) 
    sig_w = torch.tensor(0.7054) # Recalculate this too
    sig_z = torch.tensor(1.0)
    nSamples_z = torch.tensor(8)
    mu_z = torch.tensor(5)
    int_a = mu_z - 3 * sig_z
    int_b = mu_z + 3 * sig_z
    dz = (int_b - int_a) / (nSamples_z - 1)
    ci = torch.tensor(6.2)
    phi = 2 # order of Super-Gaussian

    c, n, _ = p.shape

    tileSz = torch.tensor(sz[0]//16)
    tileCt_y = sz[0] / tileSz
    tileCt_x = sz[1] / tileSz
    tileCt_yx = tileCt_y * tileCt_x
    tileSz_p = tileSz + 2 * wsz

    pd = torch.cat((p, d.unsqueeze(-1), dw.unsqueeze(-1)), 2)
    pt = torch.zeros((c, tileCt_yx, tileMaxPts, 4))

    maxPts = torch.tensor(0)

    #
    # Assign points to overlapping 2D tiles.
    # The tiles are processed in parallel. The overlap accounts for points straddling
    # the boundary of two or more tiles.
    for i in torch.arange(0, tileCt_yx):
        xt = (i // tileCt_y).int() * tileSz - wsz
        yt = (i % tileCt_y) * tileSz - wsz

        idx = torch.logical_and( torch.logical_and(pd[:, :, 0] < xt + tileSz_p, pd[:, :, 0] >= xt),
                                 torch.logical_and(pd[:, :, 1] < yt + tileSz_p, pd[:, :, 1] >= yt))
        _, ct = torch.unique(idx.nonzero()[:, 0], return_counts=True)

        if ct.shape[0] == 0:
            continue
            
        maxPts = torch.max(torch.max(ct), maxPts)
        pt[:, i, :, :] = torch.stack([F.pad(t, (0, 0, 0, torch.abs(tileMaxPts - t.shape[0])), value=-1e10) for t in torch.split( pd[idx.nonzero(as_tuple=True)], ct.tolist())]) - torch.tensor([xt, yt, 0, 0])
    o = pt[:, :, :, 2].sort(2, descending=True)[1].unsqueeze(-1)
    s = torch.gather(pt, 2, o.expand(-1, -1, -1, 4))

    # 
    # Calculate the pixel contribution of each point 
    x, y, z = torch.meshgrid( [torch.arange(-wsz, wsz + 1).float(), 
                               torch.arange(-wsz, wsz + 1).float(),
                               torch.linspace(int_a, int_b, nSamples_z)])
    x = x[:, :, 0]
    y = y[:, :, 0]
    ox = s[:, :, :, 0].unsqueeze(-1).unsqueeze(-1)
    oy = s[:, :, :, 1].unsqueeze(-1).unsqueeze(-1)
    px = torch.ceil(ox + x)
    py = torch.ceil(oy + y)
    pz = z
    
    two = torch.tensor(2.0)
    oidx = torch.logical_or( torch.logical_or(py < 0, py >= tileSz_p), torch.logical_or(px < 0, px >= tileSz_p))
    px[oidx] = 0
    py[oidx] = 0

    R = torch.zeros((c, tileCt_yx, tileSz_p * tileSz_p), dtype=torch.float32) # The splatted label as a Gaussian
    M = torch.ones((c, tileCt_yx, tileSz_p * tileSz_p), dtype=torch.float32)  # Temporary variable for accumulating density from all points along a ray
    K = torch.zeros((c, tileCt_yx, tileSz_p * tileSz_p), dtype=torch.float32) # The weight as a Super-Gaussian

    gaussianIntegral_z = torch.erf( -mu_z / (torch.sqrt(two) * sig_z)) - torch.erf((pz - mu_z) / (torch.sqrt(two) * sig_z))

    # The splatted value depends on the z-order of the points 
    # As a result, the points within each tile are splatted in serial
    for i in range(maxPts): 
        x = px[:, :, i, :, :]
        y = py[:, :, i, :, :]

        # w_i is the Super-Gaussian data weight
        w_i = (torch.exp(- torch.pow(((x - ox[:, :, i, :, :]).square() + (y - oy[:, :, i, :, :]).square())
                                    / (2 * sig_w.square()), phi)) * s[:, :, i, 3].unsqueeze(-1).unsqueeze(-1)).view(c, tileCt_yx, (wsz * 2 + 1) * (wsz * 2 + 1))
        c_i = ci * torch.exp(-((x - ox[:, :, i, :, :]).square() + (y - oy[:, :, i, :, :]).square()) / (2 * sig_d.square()))

        g_i = ci * torch.exp( -( (((x - ox[:, :, i, :, :]).square() / (2 * sig_d.square())) + # Gaussian density
                                  ((y - oy[:, :, i, :, :]).square() / (2 * sig_d.square())) ).unsqueeze(-1).expand(-1, -1, -1, -1, nSamples_z) +
                                 ((z - mu_z).square() / (2 * sig_z.square())) ))
        t_i = torch.exp( (sig_z * c_i.unsqueeze(-1).expand(-1, -1, -1, -1, nSamples_z)) * torch.sqrt(PI/two) * gaussianIntegral_z ) # Transmittance
        e_i = t_i[:, :, :, :, -1].view(c, tileCt_yx, (wsz * 2 + 1) * (wsz * 2 + 1)) # Transmittance at point where ray exits Gaussian boundary in z
        s_i = (torch.sum( g_i * t_i * dz, -1) * s[:, :, i, 2].unsqueeze(-1).unsqueeze(-1)).view(c, tileCt_yx, (wsz * 2 + 1) * (wsz * 2 + 1)) # Transmitted albedo (depth label value)
        
        idx = torch.reshape((y * tileSz_p + x).long(), (c, tileCt_yx, (wsz * 2 + 1) * (wsz * 2 + 1)))

        m_i = torch.gather(M, 2, idx)
        R.scatter_add_(2, idx, s_i * m_i)
        K.scatter_add_(2, idx, w_i * m_i)
        M = M.scatter(2, idx, e_i * m_i)

    Q = torch.zeros((c, sz[0], sz[1]), dtype=torch.float64)

    # Reshape arrays to raster size
    R = torch.reshape(R, (c, tileCt_y, tileCt_x, tileSz_p, tileSz_p))[:, :, :, wsz:-wsz, wsz:-wsz]
    R = R.permute(0, 2, 3, 1, 4).contiguous().view(c, tileCt_y, tileSz, -1).view(c, sz[0], -1)
    K = torch.reshape(K, (c, tileCt_y, tileCt_x, tileSz_p, tileSz_p))[:, :, :, wsz:-wsz, wsz:-wsz]
    K = K.permute(0, 2, 3, 1, 4).contiguous().view(c, tileCt_y, tileSz, -1).view(c, sz[0], -1)

    return (R, K)
