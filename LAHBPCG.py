
import math
import torch
import torch.nn as nn

class PCG:
    """A differentiable and batch-able version of the LAHBPCG solver (Locally Adaptive Hierarchical Basis
    Preconditing, Richard Szeliski 2006). The code is based on the C++ version of the solver from the 
    ImageStack library (https://github.com/s-gupta/rgbdutils/tree/master/imagestack)"""

    def __init__(self, channels, height, width):

        (self.channels, self.height, self.width) = channels, height, width

        self.f = torch.zeros((self.channels, self.height, self.width), dtype=torch.float32)
        self.max_length = self.height * self.width
        self.index_map = []
        self.idx0 = []
        self.idx1 = []
        self.idx2 = []
        self.idx3 = []
        self.idx4 = []
        self.idx5 = []
        self.idx6 = []
        self.hb_b1 = []
        self.hb_b2 = []
        self.hb_b3 = []
        self.hb_b4 = []
        self.oddlevel = []
        self.nw_inrange = []
        self.ws_inrange = []
        self.se_inrange = []
        self.en_inrange = []
        self.ns_inrange = []
        self.we_inrange = []
        self.n_ind = []
        self.w_ind = []
        self.s_ind = []
        self.e_ind = []
        self.ss = []
        self.sn = []
        self.sw = []
        self.se = []

        self.pady = nn.ConstantPad2d((0, 0, 0, 1), 0)
        self.padx = nn.ConstantPad2d((0, 1, 0, 0), 0)

        self.__RBBmaps()
        self.__init_indices()
        

    def __ind2xy(self, index):
        x = int(index // self.height)
        y = int(index % self.height)
        return (x, y)


    def __ind2xy_tensor(self, index):
        x = (index // self.height).long()
        y = (index % self.height).long()
        return (x, y)


    def __var_indices_tensor(self, x, y):
        v = x * self.height + y
        v[torch.logical_or(x < 0, x >= self.width)] = self.max_length
        v[torch.logical_or(y < 0, y >= self.width)] = self.max_length
        return v


    def __var_indices(self, x, y):
        if x < 0 or x >= self.width:
            return self.max_length
        if y < 0 or y >= self.height:
            return self.max_length

        return (x * self.height + y)


    def __RBBmaps(self):
        
        numOctaves = math.ceil(math.log(min(self.width, self.height))/math.log(2.0))
        by = 0
        bx = 0
        a = 1
        valid = []
        new_valid = []

        for i in range(numOctaves):
            indices1 = []
            indices2 = []

            if not valid:
                for x in range(self.width):
                    for y in range(self.height):
                        if ((x + y) % (2 * a) == (by + bx + a) % (2 * a)):
                            indices1.append(x * self.height + y)
                        else:
                            valid.append(x * self.height + y)
                    
                if not indices1:
                    break
                self.index_map.append(indices1)
            
                new_valid = []
                for v in valid:
                    x, y = self.__ind2xy(v)

                    if (y % (2 * a) == (by + a) % (2 * a)):
                        indices2.append(v)
                    else:
                        new_valid.append(v)

                valid, new_valid = new_valid, valid

                if not indices2:
                    break
                self.index_map.append(indices2)

            else:
                new_valid = []
                for v in valid:
                    x, y = self.__ind2xy(v)

                    if ((x + y) % (2 * a) == (bx + by + a) % (2 * a)):
                        indices1.append(v)
                    else:
                        new_valid.append(v)

                valid, new_valid = new_valid, valid

                if not indices1:
                    break
                self.index_map.append(indices1)

                new_valid = []
                for v in valid:
                    x, y = self.__ind2xy(v)
                    if ((y) % (2 * a) == (by + a) % (2 * a)):
                        indices2.append(v)
                    else:
                        new_valid.append(v)

                valid, new_valid = new_valid, valid

                if not indices2:
                    break
                self.index_map.append(indices2)

            a *= 2


    def __init_indices(self):

        for k in range(len(self.index_map)):
            odd_level = (k + 1) % 2
            stride = 1 << int(k//2)

            if odd_level:
                dn1 = stride
                dn2 = stride * self.height
            else:
                dn1 = stride * (self.height - 1)
                dn2 = stride * (self.height + 1)

            idx0 = torch.tensor(self.index_map[k])
            idx1 = (idx0 < dn1)  * (self.max_length + idx0 - dn1) + \
                   (idx0 >= dn1) * ((idx0 - dn1) % self.max_length)
            idx2 = (idx0 < dn2)  * (self.max_length + idx0 - dn2) + \
                   (idx0 >= dn2) * ((idx0 - dn2) % self.max_length)
            idx3 = (idx0 - dn1) % self.max_length
            idx4 = (idx0 - dn2) % self.max_length
            idx5 = (idx0 + dn1) % self.max_length
            idx6 = (idx0 + dn2) % self.max_length

            hb_b1 = idx0 + dn1 < self.max_length
            hb_b2 = idx0 + dn2 < self.max_length
            hb_b3 = idx0 >= dn1
            hb_b4 = idx0 >= dn2
            
            x, y = self.__ind2xy_tensor(idx0)
            n_ind = odd_level * self.__var_indices_tensor(x, y - stride).clone().detach() + \
                    (1 - odd_level) * self.__var_indices_tensor(x - stride, y - stride).clone().detach()
            w_ind = odd_level * self.__var_indices_tensor(x - stride, y).clone().detach() + \
                    (1 - odd_level) * self.__var_indices_tensor(x - stride, y + stride).clone().detach()
            s_ind = odd_level * self.__var_indices_tensor(x, y + stride).clone().detach() + \
                    (1 - odd_level) * self.__var_indices_tensor(x + stride, y + stride).clone().detach()
            e_ind = odd_level * self.__var_indices_tensor(x + stride, y).clone().detach() + \
                    (1 - odd_level) * self.__var_indices_tensor(x + stride, y - stride).clone().detach()

            ns_inrange = torch.logical_and(n_ind < self.max_length, s_ind < self.max_length)
            we_inrange = torch.logical_and(w_ind < self.max_length, e_ind < self.max_length)
            nw_inrange = torch.logical_and(n_ind < self.max_length, w_ind < self.max_length)
            ws_inrange = torch.logical_and(w_ind < self.max_length, s_ind < self.max_length)
            se_inrange = torch.logical_and(e_ind < self.max_length, s_ind < self.max_length)
            en_inrange = torch.logical_and(e_ind < self.max_length, n_ind < self.max_length)
            
            self.idx0.append(idx0)
            self.idx1.append(idx1)
            self.idx2.append(idx2)
            self.idx3.append(idx3)
            self.idx4.append(idx4)
            self.idx5.append(idx5)
            self.idx6.append(idx6)
            self.hb_b1.append(hb_b1)
            self.hb_b2.append(hb_b2)
            self.hb_b3.append(hb_b3)
            self.hb_b4.append(hb_b4)
            self.oddlevel.append(odd_level)
            self.nw_inrange.append(nw_inrange)
            self.ws_inrange.append(ws_inrange)
            self.se_inrange.append(se_inrange)
            self.en_inrange.append(en_inrange)
            self.ns_inrange.append(ns_inrange)
            self.we_inrange.append(we_inrange)
            self.n_ind.append(n_ind)
            self.w_ind.append(w_ind)
            self.s_ind.append(s_ind)
            self.e_ind.append(e_ind)


    def __construct_preconditioner(self):

        for k in range(len(self.index_map)):
            x, y = self.__ind2xy_tensor(self.idx0[k])
            ss = -self.AN[:, y, x] / self.AD[:, y, x]
            se = -self.AW[:, y, x] / self.AD[:, y, x]
            
            x1, y1 = self.__ind2xy_tensor(self.idx1[k])
            sn = -self.AN[:, y1, x1] / self.AD[:, y, x]

            x1, y1 = self.__ind2xy_tensor(self.idx2[k])
            sw = -self.AW[:, y1, x1] / self.AD[:, y, x]

            AD_old = self.AD[:, y, x]

            AN_tmp = torch.zeros((self.channels, self.height, self.width), dtype=torch.float64)
            AW_tmp = torch.zeros((self.channels, self.height, self.width), dtype=torch.float64)

            x1, y1 = self.__ind2xy_tensor(self.idx3[k])
            self.AD[:, y1, x1] += self.AN[:, y1, x1] * sn

            x1, y1 = self.__ind2xy_tensor(self.idx4[k])
            self.AD[:, y1, x1] += self.AW[:, y1, x1] * sw

            x1, y1 = self.__ind2xy_tensor(self.idx5[k])
            self.AD[:, y1, x1] += self.AN[:, y, x] * ss

            x1, y1 = self.__ind2xy_tensor(self.idx6[k])
            self.AD[:, y1, x1] += self.AW[:, y, x] * se

            self.ss.append(ss)
            self.sw.append(sw)
            self.sn.append(sn)
            self.se.append(se)
                
            ns_weight = self.oddlevel[k] * (-AD_old * sn * ss) + \
                        (1 - self.oddlevel[k]) * (-AD_old * sw * se)
            n_x, n_y = self.__ind2xy_tensor(self.n_ind[k][self.ns_inrange[k]])
            s_x, s_y = self.__ind2xy_tensor(self.s_ind[k][self.ns_inrange[k]])

            self.AD[:, n_y, n_x] += ns_weight[:, self.ns_inrange[k]]
            self.AD[:, s_y, s_x] += ns_weight[:, self.ns_inrange[k]]

            we_weight = self.oddlevel[k] * (-AD_old * sw * se) + \
                        (1 - self.oddlevel[k]) * (-AD_old * sn * ss)
            w_x, w_y = self.__ind2xy_tensor(self.w_ind[k][self.we_inrange[k]])
            e_x, e_y = self.__ind2xy_tensor(self.e_ind[k][self.we_inrange[k]])
            self.AD[:, w_y, w_x] += we_weight[:, self.we_inrange[k]]
            self.AD[:, e_y, e_x] += we_weight[:, self.we_inrange[k]]

            n_x, n_y = self.__ind2xy_tensor(self.n_ind[k])
            s_x, s_y = self.__ind2xy_tensor(self.s_ind[k])
            w_x, w_y = self.__ind2xy_tensor(self.w_ind[k])
            e_x, e_y = self.__ind2xy_tensor(self.e_ind[k])

            nw_weight = AD_old * sn * sw
            y_ = self.oddlevel[k] * w_y[self.nw_inrange[k]] + (1 - self.oddlevel[k]) * n_y[self.nw_inrange[k]]
            x_ = self.oddlevel[k] * w_x[self.nw_inrange[k]] + (1 - self.oddlevel[k]) * n_x[self.nw_inrange[k]]
            AN_tmp[:, y_, x_] -= nw_weight[:, self.nw_inrange[k]]

            ws_weight = self.oddlevel[k] * (AD_old * sw * ss) + (1 - self.oddlevel[k]) * (AD_old * sn * se)
            y_ = w_y[self.ws_inrange[k]]
            x_ = w_x[self.ws_inrange[k]]
            AW_tmp[:, y_, x_] -= ws_weight[:, self.ws_inrange[k]]

            se_weight = AD_old * se * ss
            y_ = self.oddlevel[k] * s_y[self.se_inrange[k]] + (1 - self.oddlevel[k]) * e_y[self.se_inrange[k]]
            x_ = self.oddlevel[k] * s_x[self.se_inrange[k]] + (1 - self.oddlevel[k]) * e_x[self.se_inrange[k]]
            AN_tmp[:, y_, x_] -= se_weight[:, self.se_inrange[k]]

            en_weight = self.oddlevel[k] * (AD_old * se * sn) + (1 - self.oddlevel[k]) * (AD_old * ss * sw)
            y_ = n_y[self.en_inrange[k]]
            x_ = n_x[self.en_inrange[k]]
            AW_tmp[:, y_, x_] -= en_weight[:, self.en_inrange[k]]

            total = nw_weight + ws_weight + se_weight + en_weight
            total_idx = torch.abs(total) > 1e-10
            nw_weight[total_idx] /= total[total_idx]
            ws_weight[total_idx] /= total[total_idx]
            se_weight[total_idx] /= total[total_idx]
            en_weight[total_idx] /= total[total_idx]
            
            sN = 2.0
            dist_weight = sN * (ns_weight + we_weight)

            ns_or_we = torch.logical_or(self.ns_inrange[k], self.we_inrange[k])

            y_ = self.oddlevel[k] * w_y[torch.logical_and(self.nw_inrange[k], ns_or_we)] + \
                 (1 - self.oddlevel[k]) * n_y[torch.logical_and(self.nw_inrange[k], ns_or_we)]
            x_ = self.oddlevel[k] * w_x[torch.logical_and(self.nw_inrange[k], ns_or_we)] + \
                 (1 - self.oddlevel[k]) * n_x[torch.logical_and(self.nw_inrange[k], ns_or_we)]
            AN_tmp[:, y_, x_] += nw_weight[:, torch.logical_and(self.nw_inrange[k], ns_or_we)] * \
                                    dist_weight[:, torch.logical_and(self.nw_inrange[k], ns_or_we)]

            y_ = w_y[torch.logical_and(self.ws_inrange[k], ns_or_we)] 
            x_ = w_x[torch.logical_and(self.ws_inrange[k], ns_or_we)]
            AW_tmp[:, y_, x_] += ws_weight[:, torch.logical_and(self.ws_inrange[k], ns_or_we)] * \
                                    dist_weight[:, torch.logical_and(self.ws_inrange[k], ns_or_we)]

            y_ = self.oddlevel[k] * s_y[torch.logical_and(self.se_inrange[k], ns_or_we)] + \
                 (1 - self.oddlevel[k]) * e_y[torch.logical_and(self.se_inrange[k], ns_or_we)]
            x_ = self.oddlevel[k] * s_x[torch.logical_and(self.se_inrange[k], ns_or_we)] + \
                 (1 - self.oddlevel[k]) * e_x[torch.logical_and(self.se_inrange[k], ns_or_we)]
            AN_tmp[:, y_, x_] += se_weight[:, torch.logical_and(self.se_inrange[k], ns_or_we)] * \
                                    dist_weight[:, torch.logical_and(self.se_inrange[k], ns_or_we)]

            y_ = n_y[torch.logical_and(self.en_inrange[k], ns_or_we)] 
            x_ = n_x[torch.logical_and(self.en_inrange[k], ns_or_we)]
            AW_tmp[:, y_, x_] += en_weight[:, torch.logical_and(self.en_inrange[k], ns_or_we)] * \
                                    dist_weight[:, torch.logical_and(self.en_inrange[k], ns_or_we)]

            y_ = n_y[torch.logical_and(self.nw_inrange[k], ns_or_we)]
            x_ = n_x[torch.logical_and(self.nw_inrange[k], ns_or_we)]
            self.AD[:, y_, x_] -= nw_weight[:, torch.logical_and(self.nw_inrange[k], ns_or_we)] * \
                                  dist_weight[:, torch.logical_and(self.nw_inrange[k], ns_or_we)]

            y_ = w_y[torch.logical_and(self.nw_inrange[k], ns_or_we)]
            x_ = w_x[torch.logical_and(self.nw_inrange[k], ns_or_we)]
            self.AD[:, y_, x_] -= nw_weight[:, torch.logical_and(self.nw_inrange[k], ns_or_we)] * \
                                  dist_weight[:, torch.logical_and(self.nw_inrange[k], ns_or_we)]

            y_ = w_y[torch.logical_and(self.ws_inrange[k], ns_or_we)]
            x_ = w_x[torch.logical_and(self.ws_inrange[k], ns_or_we)]
            self.AD[:, y_, x_] -= ws_weight[:, torch.logical_and(self.ws_inrange[k], ns_or_we)] * \
                                  dist_weight[:, torch.logical_and(self.ws_inrange[k], ns_or_we)]

            y_ = s_y[torch.logical_and(self.ws_inrange[k], ns_or_we)]
            x_ = s_x[torch.logical_and(self.ws_inrange[k], ns_or_we)]
            self.AD[:, y_, x_] -= ws_weight[:, torch.logical_and(self.ws_inrange[k], ns_or_we)] * \
                                  dist_weight[:, torch.logical_and(self.ws_inrange[k], ns_or_we)]

            y_ = s_y[torch.logical_and(self.se_inrange[k], ns_or_we)]
            x_ = s_x[torch.logical_and(self.se_inrange[k], ns_or_we)]
            self.AD[:, y_, x_] -= se_weight[:, torch.logical_and(self.se_inrange[k], ns_or_we)] * \
                                  dist_weight[:, torch.logical_and(self.se_inrange[k], ns_or_we)]

            y_ = e_y[torch.logical_and(self.se_inrange[k], ns_or_we)]
            x_ = e_x[torch.logical_and(self.se_inrange[k], ns_or_we)]
            self.AD[:, y_, x_] -= se_weight[:, torch.logical_and(self.se_inrange[k], ns_or_we)] * \
                                  dist_weight[:, torch.logical_and(self.se_inrange[k], ns_or_we)]

            y_ = n_y[torch.logical_and(self.en_inrange[k], ns_or_we)]
            x_ = n_x[torch.logical_and(self.en_inrange[k], ns_or_we)]
            self.AD[:, y_, x_] -= en_weight[:, torch.logical_and(self.en_inrange[k], ns_or_we)] * \
                                  dist_weight[:, torch.logical_and(self.en_inrange[k], ns_or_we)]

            y_ = e_y[torch.logical_and(self.en_inrange[k], ns_or_we)]
            x_ = e_x[torch.logical_and(self.en_inrange[k], ns_or_we)]
            self.AD[:, y_, x_] -= en_weight[:, torch.logical_and(self.en_inrange[k], ns_or_we)] * \
                                  dist_weight[:, torch.logical_and(self.en_inrange[k], ns_or_we)]

            self.AN = AN_tmp
            self.AW = AW_tmp
            
            
    def __Ax(self, im):

        channels, height, width = im.shape

        a1 = 0
        a2 = self.sx[:, :, 0] + self.sx[:, :, 1] + self.w[:, :, 0]
        a3 = -self.sx[:, :, 1]
        self.f[:, :, 0] = a2 * im[:, :, 0] + a3 * im[:, :, 1]

        a1 = -self.sx[:, :, 1:-1]
        a2 = self.sx[:, :, 1:-1] + self.sx[:, :, 2:] + self.w[:, :, 1:-1]
        a3 = -self.sx[:, :, 2:]
        self.f[:, :, 1:-1] = a1 * im[:, :, :-2] + a2 * im[:, :, 1:-1] + a3 * im[:, :, 2:]

        a1 = -self.sx[:, :, -1]
        a2 = self.sx[:, :, -1] + self.w[:, :, -1]
        a3 = 0
        self.f[:, :, -1] = a1 * im[:, :, -2] + a2 * im[:, :, -1]

        a1 = 0
        a2 = self.sy[:, 0, :] + self.sy[:, 1, :]
        a3 = -self.sy[:, 1, :]
        self.f[:, 0, :] += a2 * im[:, 0, :] + a3 * im[:, 1, :]

        a1 = -self.sy[:, 1:-1, :]
        a2 = self.sy[:, 1:-1, :] + self.sy[:, 2:, :]
        a3 = -self.sy[:, 2:, :]
        self.f[:, 1:-1, :] += a1 * im[:, :-2, :] + a2 * im[:, 1:-1, :] + a3 * im[:, 2:, :]

        a1 = -self.sy[:, -1, :]
        a2 = self.sy[:, -1, :]
        a3 = 0
        self.f[:, -1, :] += a1 * im[:, -2, :] + a2 * im[:, -1, :]
        return self.f.clone()

    
    def __hb_precondition(self, r):
        hbRes = r.clone()

        for k in range(len(self.index_map)):
            x, y = self.__ind2xy_tensor(self.idx0[k][self.hb_b1[k]])
            x1, y1 = self.__ind2xy_tensor(self.idx5[k][self.hb_b1[k]])            
            hbRes[:, y1, x1] += hbRes[:, y, x].clone() * self.ss[k][:, self.hb_b1[k]]

            x, y = self.__ind2xy_tensor(self.idx0[k][self.hb_b2[k]])
            x1, y1 = self.__ind2xy_tensor(self.idx6[k][self.hb_b2[k]])            
            hbRes[:, y1, x1] += hbRes[:, y, x].clone() * self.se[k][:, self.hb_b2[k]]

            x, y = self.__ind2xy_tensor(self.idx0[k][self.hb_b3[k]])
            x1, y1 = self.__ind2xy_tensor(self.idx3[k][self.hb_b3[k]])            
            hbRes[:, y1, x1] += hbRes[:, y, x].clone() * self.sn[k][:, self.hb_b3[k]]

            x, y = self.__ind2xy_tensor(self.idx0[k][self.hb_b4[k]])
            x1, y1 = self.__ind2xy_tensor(self.idx4[k][self.hb_b4[k]])            
            hbRes[:, y1, x1] += hbRes[:, y, x].clone() * self.sw[k][:, self.hb_b4[k]]

        hbRes /= self.AD

        for k in range(len(self.index_map) - 1, -1, -1):

            x, y = self.__ind2xy_tensor(self.idx0[k][self.hb_b1[k]])
            x1, y1 = self.__ind2xy_tensor(self.idx5[k][self.hb_b1[k]])            
            hbRes[:, y, x] += hbRes[:, y1, x1].clone() * self.ss[k][:, self.hb_b1[k]]

            x, y = self.__ind2xy_tensor(self.idx0[k][self.hb_b2[k]])
            x1, y1 = self.__ind2xy_tensor(self.idx6[k][self.hb_b2[k]])            
            hbRes[:, y, x] += hbRes[:, y1, x1].clone() * self.se[k][:, self.hb_b2[k]]

            x, y = self.__ind2xy_tensor(self.idx0[k][self.hb_b3[k]])
            x1, y1 = self.__ind2xy_tensor(self.idx3[k][self.hb_b3[k]])            
            hbRes[:, y, x] += hbRes[:, y1, x1].clone() * self.sn[k][:, self.hb_b3[k]]

            x, y = self.__ind2xy_tensor(self.idx0[k][self.hb_b4[k]])
            x1, y1 = self.__ind2xy_tensor(self.idx4[k][self.hb_b4[k]])            
            hbRes[:, y, x] += hbRes[:, y1, x1].clone() * self.sw[k][:, self.hb_b4[k]]

        return hbRes


    def set_constraints(self, d, gx, gy, w, sx, sy):
        """ Set constraints for dense diffusion from a sparse set of points

        Args:
          d (c x h x w tensor): sparse labels
          gx (c x h x w tensor): target gradients in x (for smooth diffusion, these are zero)
          gy (c x h x w tensor): target gradients in y (for smooth diffusion, these are zero)
          w (c x h x w tensor): data weights for sparse labels
          sx (c x h x w tensor): smoothness weights in x
          sy (c x h x w tensor): smoothness weights in y
        
        Returns:
          None
        """

        self.sx = sx
        self.sy = sy
        self.w = w
        self.ss = []
        self.sn = []
        self.sw = []
        self.se = []

        sy_r = self.pady(sy)[:, 1:, :]
        gy_r = self.pady(gy)[:, 1:, :]
        sx_r = self.padx(sx)[:, :, 1:]
        gx_r = self.padx(gx)[:, :, 1:]

        self.AN = -sy_r
        self.AW = -sx_r
        self.AD = sx + sx_r + w + sy + sy_r
        self.b = gy * sy - (gy_r * sy_r) + gx * sx - (gx_r * sx_r) + w * d

        self.__construct_preconditioner()

    def solve(self, guess, max_iter, tol):
        """ Optimize a dense output from sparse labels based on set consttaints.
        Optimization stops when either maximum iterations or minimum error threshold is reached.

        Args:
           guess (c x h x w tensor): initial estimate 
           max_iter (int): maximum number of optimization iterations
           tol (float): minimum error threshold

        Returns:
           tensor (c x h x w): dense diffusion output
        """

        lguess = [guess]
        lr = [self.b - self.__Ax(guess)]
                
        dr = self.__hb_precondition(lr[0])

        delta = torch.sum(lr[0] * dr)
        epsilon = tol * tol * delta

        j = 0
        for i in range(max_iter):
            wr = self.__Ax(dr)
            alpha = delta / torch.sum(dr * wr) 

            lguess.append( lguess[i] + dr  * alpha)
            lr.append( lr[i] - wr * alpha)

            s = self.__hb_precondition(lr[ i + 1])
            delta_old = delta
            delta = torch.sum(lr[i + 1] * s) 

            dr = s + dr * (delta / delta_old)
            j = j + 1;
            
        return lguess[j]
