#
# STELLA stencils
#
import logging

import numpy as np

from gridtools.stencil  import Stencil, MultiStageStencil



class FastWavesUV (MultiStageStencil):
    def __init__ (self, domain, dt_small=10.0/3.0, dlat=0.02, flat_limit=11):
        super ( ).__init__ ( )

        # self.domain = domain
        # self.set_halo(halo)
        # self.set_k_direction('forward')

        # Constants
        self.earth_radius = 6371.229e3
        self.gravity      = 9.80665
        self.dt_small     = dt_small
        self.dlat         = dlat
        self.edadlat      = self.earth_radius / (self.dlat*np.pi / 180.0)
        self.flat_limit   = flat_limit

        # # Fields to be further initialized
        # self.utens_stage = np.zeros (domain, dtype=np.float64)
        # self.vtens_stage = np.zeros (domain, dtype=np.float64)
        # self.u_pos     = np.zeros (domain, dtype=np.float64)
        # self.v_pos     = np.zeros (domain, dtype=np.float64)
        # self.ppuv      = np.zeros (domain, dtype=np.float64)
        # self.rho       = np.zeros (domain, dtype=np.float64)
        # self.rho0      = np.zeros (domain, dtype=np.float64)
        # self.p0        = np.zeros (domain, dtype=np.float64)
        # self.hhl       = np.zeros (domain, dtype=np.float64)
        # self.wgtfac    = np.zeros (domain, dtype=np.float64)
        # self.fx        = np.zeros (domain, dtype=np.float64)
        # self.cwp       = np.zeros (domain, dtype=np.float64)
        # self.xdzdx     = np.zeros (domain, dtype=np.float64)
        # self.xdzdy     = np.zeros (domain, dtype=np.float64)
        # self.xlhsx     = np.zeros (domain, dtype=np.float64)
        # self.xlhsy     = np.zeros (domain, dtype=np.float64)
        # self.wbbctens_stage = np.zeros ((domain[0],domain[1],domain[2]+1))

        # The following are simply set to zero
        self.xrhsx     = np.zeros (domain, dtype=np.float64)
        self.xrhsy     = np.zeros (domain, dtype=np.float64)
        self.xrhsz     = np.zeros (domain, dtype=np.float64)
        self.ppgradcor_init = np.zeros (domain, dtype=np.float64)
        self.ppgradcor = np.zeros (domain, dtype=np.float64)
        self.ppgradu   = np.zeros (domain, dtype=np.float64)
        self.ppgradv   = np.zeros (domain, dtype=np.float64)

        # self._init_special()

    #
    # def _init_special(self):
    #     dx, dy, dz = [ 1./i for i in self.domain ]
    #
    #     # utens_stage, vtens_stage
    #     for p in self.get_interior_points (self.utens_stage,
    #                                        ghost_cell=[0,1,0,1]):
    #         x = dx*p[0]
    #         y = dy*p[1]
    #         z = dz*p[2]
    #         self.utens_stage[p] = 2e-4*(-1 +
    #                                     2*(2. +
    #                                        np.cos(np.pi*(x+z)) +
    #                                        np.cos(np.pi*y))/4. +
    #                                     0.1*(0.5-(np.random.random()%100)/50.))
    #         self.vtens_stage[p] = self.utens_stage[p]
    #
    #     # ppuv, rho, rho0, p0, wgtfac, fx
    #     for p in self.get_interior_points (self.ppuv, ghost_cell=[0,0,0,0]):
    #         x = dx*p[0]
    #         y = dy*p[1]
    #         self.ppuv[p] = 0.6 + (0.89-0.6)*(2.+np.cos(np.pi*(x+y)) +
    #                                          np.sin(2*np.pi*(x+y)))/4.0
    #         self.rho[p]  = 0.1 + 0.19*(2.+np.cos(np.pi*(x+y)) +
    #                                    np.sin(2*np.pi*(x+y)))/4.0
    #         self.rho0[p]   = self.rho[p] + (np.random.random()%100)/100.0*0.01
    #         self.p0[p]     = self.ppuv[p] + (np.random.random()%100)/100.0*0.01
    #         self.wgtfac[p] = self.p0[p] + (np.random.random()%100)/100.0*0.01
    #
    #     # hhl
    #     for p in self.get_interior_points (self.hhl, ghost_cell=[0,0,0,0]):
    #         if p[2] < self.flat_limit:
    #             self.hhl[p] = 0.
    #         else:
    #             r = 500. / 48. * (p[2] - 14)
    #             min = 10000 / 48. * abs(p[2] - self.domain[2])
    #             x = dx*p[0]
    #             y = dy*p[1]
    #             self.hhl[p] = min + r*(2.+np.cos(np.pi*(x+y)) +
    #                                    np.sin(2*np.pi*(x+y)))/4.0
    #
    #     # fx
    #     for p in self.get_interior_points (self.fx[0:1,:,0:1], ghost_cell=[0,0,0,0]):
    #         self.fx[p] = self.p0[p] + (np.random.random()%100)/100.0*0.01
    #
    #     # cwp, xdzdx, xdzdy, xlhsx, xlhsy, wbbctens_stage
    #     for p in self.get_interior_points (self.cwp[:,:,0:1], ghost_cell=[0,0,0,0]):
    #         self.cwp[p]   = self.p0[p] + (np.random.random()%100)/100.0*0.01
    #         self.xdzdx[p] = self.p0[p] + (np.random.random()%100)/100.0*0.01
    #         self.xdzdy[p] = self.p0[p] + (np.random.random()%100)/100.0*0.01
    #         self.xlhsx[p] = self.p0[p] + (np.random.random()%100)/100.0*0.01
    #         self.xlhsy[p] = self.p0[p] + (np.random.random()%100)/100.0*0.01
    #         self.wbbctens_stage[p] = self.p0[p] + \
    #                                  (np.random.random()%100)/100.0*0.01

#
#
#    def stage_ppgradcor_at_flat_limit (self, ppgradcor, wgtfac, ppuv):
#        #
#        # compute ppgradcor at k = self.flat_limit
#        #
#        for p in self.get_interior_points (
#                #ppgradcor[:,:,self.flat_limit:self.flat_limit+1],
#                ppgradcor[:,:,11:12],
#                ghost_cell=[0,1,0,1]):
#            ppgradcor[p] = wgtfac[p]*ppuv[p] + (1.0 - wgtfac[p]) * ppuv[p + (0,0,-1)]
#
#
#    def stage_ppgradcor_over_flat_limit (self, ppgradcor, wgtfac, ppuv):
#        #
#        # compute ppgradcor at k > self.flat_limit
#        #
#        for p in self.get_interior_points (
##                ppgradcor[:,:,self.flat_limit+1:],
#                ppgradcor[:,:,12:],
#                ghost_cell=[0,1,0,1]):
#            ppgradcor[p] = wgtfac[p]*ppuv[p] + (1.0 - wgtfac[p]) * ppuv[p + (0,0,-1)]
#            ppgradcor[p + (0,0,-1)] = ppgradcor[p] - ppgradcor[p + (0,0,-1)]


    def stage_ppgradcor_init (self, ppgradcor_init, wgtfac, ppuv):
        #
        # initialize ppgradcor values at k >= self.flat_limit
        #
        for p in self.get_interior_points (
                #ppgradcor_init[:,:,self.flat_limit:],
                ppgradcor_init[:,:,10:],
                ghost_cell=[0,1,0,1]):
            ppgradcor_init[p] = wgtfac[p]*ppuv[p] + (1.0 - wgtfac[p]) * ppuv[p + (0,0,-1)]


    def stage_ppgradcor_below_top (self, ppgradcor_init, ppgradcor, wgtfac, ppuv):
        #
        # compute ppgradcor at k > self.flat_limit and k < top
        #
        for p in self.get_interior_points (
#                ppgradcor[:,:,self.flat_limit:self.domain[2]-1],
                ppgradcor[:,:,10:79],
                ghost_cell=[0,1,0,1]):
            ppgradcor[p] = ppgradcor_init[p + (0,0,1)] - ppgradcor_init[p]


    def stage_ppgradcor_at_top (self, ppgradcor, wgtfac, ppuv):
        #
        # compute ppgradcor at k = top
        #
        for p in self.get_interior_points (
#                ppgradcor[:,:,self.domain[2]-1:],
                ppgradcor[:,:,79:],
                ghost_cell=[0,1,0,1]):
            ppgradcor[p] = wgtfac[p]*ppuv[p] + (1.0 - wgtfac[p]) * ppuv[p + (0,0,-1)]


    def stage_xrhsx (self, xrhsx, fx, rho, ppuv, utens_stage):
        for p in self.get_interior_points (
#                xrhsx[:,:,self.domain[2]-1:],
                xrhsx[:,:,79:],
                ghost_cell=[1,0,0,1]):
            xrhsx[p] = -fx[p] / (0.5*(rho[p] +rho[p + (1,0,0)])) * \
                       (ppuv[p + (1,0,0)] - ppuv[p]) + utens_stage[p]


    def stage_xrhsy (self, xrhsy, rho, ppuv, vtens_stage):
        for p in self.get_interior_points (xrhsy[:,:,79:], ghost_cell=[0,1,1,0]):
            xrhsy[p] = -self.edadlat / (0.5*(rho[p + (0,1,0)] + rho[p])) * \
                       (ppuv[p + (0,1,0)]-ppuv[p]) + vtens_stage[p]


    def stage_xrhsz (self, xrhsz, rho0, rho, cwp, p0, ppuv, wbbctens_stage):
        for p in self.get_interior_points (
#                xrhsz[:,:,self.domain[2]-1:],
                xrhsz[:,:,79:],
                ghost_cell=[0,1,0,1]):
            xrhsz[p] = rho0[p] / rho[p] * self.gravity * \
                       (1.0 - cwp[p] * (p0[p] + ppuv[p])) + \
                       wbbctens_stage[p + (0,0,1)]


    def stage_ppgrad_at_flat_limit (self, ppgradu, ppgradv, ppuv, ppgradcor, hhl):
        # k < self.flat_limit
#        for p in self.get_interior_points (ppgradu[:,:,:self.flat_limit]):
        for p in self.get_interior_points (ppgradu[:,:,:11]):
            ppgradu[p] = ppuv[p + (1,0,0)] - ppuv[p]
            ppgradv[p] = ppuv[p + (0,1,0)] - ppuv[p]


    def stage_ppgrad_over_flat_limit (self, ppgradu, ppgradv, ppuv, ppgradcor, hhl):
        # k >= self.flat_limit
#        for p in self.get_interior_points (ppgradu[:,:,self.flat_limit:self.domain[2]-1]):
        for p in self.get_interior_points (ppgradu[:,:,11:79]):
            ppgradu[p] = (ppuv[p + (1,0,0)]-ppuv[p]) + (ppgradcor[p + (1,0,0)] + ppgradcor[p])* 0.5 * ((hhl[p + (0,0,1)] + hhl[p]) - (hhl[p + (1,0,1)]+hhl[p + (1,0,0)])) / ((hhl[p + (0,0,1)] - hhl[p]) + (hhl[p + (1,0,1)] - hhl[p + (1,0,0)]))
            ppgradv[p] = (ppuv[p + (0,1,0)]-ppuv[p]) + (ppgradcor[p + (0,1,0)] + ppgradcor[p])* 0.5 * ((hhl[p + (0,0,1)] + hhl[p]) - (hhl[p + (0,1,1)]+hhl[p + (0,1,0)])) / ((hhl[p + (0,0,1)] - hhl[p]) + (hhl[p + (0,1,1)] - hhl[p + (0,1,0)]))



    def stage_uv (self,
                  u_out, v_out,
                  u_pos, v_pos,
                  fx, rho,
                  utens_stage, vtens_stage,
                  ppgradu, ppgradv,
                  xlhsx, xlhsy,
                  xdzdx, xdzdy,
                  xrhsx, xrhsy, xrhsz):

        for p in self.get_interior_points (u_out[:,:,:79]):
            rhou = fx[p] / (0.5*(rho[p + (1,0,0)] + rho[p]))
            rhov = self.edadlat / (0.5*(rho[p + (0,1,0)] + rho[p]))
            u_out[p] = u_pos[p] + (utens_stage[p] - ppgradu[p]*rhou)*self.dt_small
            v_out[p] = v_pos[p] + (vtens_stage[p] - ppgradv[p]*rhov)*self.dt_small


    def stage_uv_boundary (self,
                  u_out, v_out,
                  u_pos, v_pos,
                  fx, rho,
                  utens_stage, vtens_stage,
                  ppgradu, ppgradv,
                  xlhsx, xlhsy,
                  xdzdx, xdzdy,
                  xrhsx, xrhsy, xrhsz):

#        for p in self.get_interior_points (u_out[:,:,self.domain[2]-1:]):
        for p in self.get_interior_points (u_out[:,:,79:]):
            bott_u = xlhsx[p] * xdzdx[p] * (0.5*(xrhsz[p + (1,0,0)]+xrhsz[p]) - xdzdx[p] * xrhsx[p] - 0.5*(0.5*(xdzdy[p + (1,-1,0)]+xdzdy[p + (1,0,0)]) + 0.5*(xdzdy[p + (0,-1,0)]+xdzdy[p])) * 0.5*(0.5*(xrhsy[p + (1,-1,0)]+xrhsy[p + (1,0,0)]) + 0.5*(xrhsy[p + (0,-1,0)]+xrhsy[p]))) + xrhsx[p]
            bott_v = xlhsy[p] * xdzdy[p] * (0.5*(xrhsz[p + (0,1,0)]+xrhsz[p]) - xdzdy[p] * xrhsy[p] - 0.5*(0.5*(xdzdx[p + (-1,1,0)]+xdzdx[p + (0,1,0)]) + 0.5*(xdzdx[p + (-1,0,0)]+xdzdx[p])) * 0.5*(0.5*(xrhsx[p + (-1,1,0)]+xrhsx[p + (0,1,0)]) + 0.5*(xrhsx[p + (-1,0,0)]+xrhsx[p]))) + xrhsy[p]
            u_out[p] = u_pos[p] + bott_u*self.dt_small
            v_out[p] = v_pos[p] + bott_v*self.dt_small


    @Stencil.kernel
    def kernel (self, u_pos, v_pos,
                utens_stage, vtens_stage,
                ppuv,
                rho, rho0,
                p0,
                hhl,
                wgtfac,
                fx,
                cwp,
                xdzdx ,xdzdy,
                xlhsx,
                xlhsy,
                wbbctens_stage,
                out_u, out_v):

#        self.stage_ppgradcor_at_flat_limit (ppgradcor=self.ppgradcor,
#                                            wgtfac=self.wgtfac,
#                                            ppuv=self.ppuv)
#        self.stage_ppgradcor_over_flat_limit (ppgradcor=self.ppgradcor,
#                                              wgtfac=self.wgtfac,
#                                              ppuv=self.ppuv)
        self.stage_ppgradcor_init (ppgradcor_init=self.ppgradcor_init,
                                   wgtfac=wgtfac,
                                   ppuv=ppuv)
        self.stage_ppgradcor_below_top (ppgradcor_init=self.ppgradcor_init,
                                        ppgradcor=self.ppgradcor,
                                        wgtfac=wgtfac,
                                        ppuv=ppuv)
        self.stage_ppgradcor_at_top (ppgradcor=self.ppgradcor,
                                     wgtfac=wgtfac,
                                     ppuv=ppuv)
        self.stage_xrhsx (xrhsx=self.xrhsx,
                          fx=fx,
                          rho=rho,
                          ppuv=ppuv,
                          utens_stage=utens_stage)
        self.stage_xrhsy (xrhsy=self.xrhsy,
                          rho=rho,
                          ppuv=ppuv,
                          vtens_stage=vtens_stage)
        self.stage_xrhsz (xrhsz=self.xrhsz,
                          rho0=rho0,
                          rho=rho,
                          cwp=cwp,
                          p0=p0,
                          ppuv=ppuv,
                          wbbctens_stage=wbbctens_stage)
        self.stage_ppgrad_at_flat_limit (ppgradu=self.ppgradu,
                                         ppgradv=self.ppgradv,
                                         ppuv=ppuv,
                                         ppgradcor=self.ppgradcor,
                                         hhl=hhl)
        self.stage_ppgrad_over_flat_limit (ppgradu=self.ppgradu,
                                           ppgradv=self.ppgradv,
                                           ppuv=ppuv,
                                           ppgradcor=self.ppgradcor,
                                           hhl=hhl)
        self.stage_uv (u_out=out_u, v_out=out_v,
                       fx=fx,
                       rho=rho,
                       u_pos=u_pos,
                       v_pos=v_pos,
                       utens_stage=utens_stage,
                       vtens_stage=vtens_stage,
                       ppgradu=self.ppgradu,
                       ppgradv=self.ppgradv,
                       xlhsx=xlhsx,
                       xlhsy=xlhsy,
                       xdzdx=xdzdx,
                       xdzdy=xdzdy,
                       xrhsx=self.xrhsx,
                       xrhsy=self.xrhsy,
                       xrhsz=self.xrhsz)
        self.stage_uv_boundary (u_out=out_u, v_out=out_v,
                                fx=fx,
                                rho=rho,
                                u_pos=u_pos,
                                v_pos=v_pos,
                                utens_stage=utens_stage,
                                vtens_stage=vtens_stage,
                                ppgradu=self.ppgradu,
                                ppgradv=self.ppgradv,
                                xlhsx=xlhsx,
                                xlhsy=xlhsy,
                                xdzdx=xdzdx,
                                xdzdy=xdzdy,
                                xrhsx=self.xrhsx,
                                xrhsy=self.xrhsy,
                                xrhsz=self.xrhsz)
        # u_out = u_in
        # v_out = v_in
