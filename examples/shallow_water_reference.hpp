/*
  GridTools Libraries

  Copyright (c) 2017, ETH Zurich and MeteoSwiss
  All rights reserved.

  Redistribution and use in source and binary forms, with or without
  modification, are permitted provided that the following conditions are
  met:

  1. Redistributions of source code must retain the above copyright
  notice, this list of conditions and the following disclaimer.

  2. Redistributions in binary form must reproduce the above copyright
  notice, this list of conditions and the following disclaimer in the
  documentation and/or other materials provided with the distribution.

  3. Neither the name of the copyright holder nor the names of its
  contributors may be used to endorse or promote products derived from
  this software without specific prior written permission.

  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
  A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
  HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
  SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
  LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
  DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
  THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
  (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
  OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

  For information: http://eth-cscs.github.io/gridtools/
*/
#pragma once
#include <gridtools.hpp>

/**
   @file
   reference implementation of the shallow_water example.
   For an exhaustive description of the shallow water problem refer to:
   http://www.mathworks.ch/moler/exm/chapters/water.pdf

*/
using namespace gridtools;

namespace {
    template < int exponent >
    static constexpr float_type pow(float_type const &x) {
        return gridtools::gt_pow< exponent >::apply(x);
    }
}

template < typename StorageType, uint_t DimI, uint_t DimJ >
struct shallow_water_reference {

    typedef StorageType storage_type;
    static constexpr uint_t strides[2] = {DimI, 1};
    static constexpr uint_t size = DimI * DimJ;
    static constexpr uint_t ip1 = strides[0];
    static constexpr uint_t jp1 = strides[1];
    static constexpr uint_t im1 = -strides[0];
    static constexpr uint_t jm1 = -strides[1];

    typename storage_type::storage_info_t solution_meta;
    typedef typename storage_type::data_store_t data_store_t;
    storage_type solution;
    float_type u_array[size];
    float_type v_array[size];
    float_type h_array[size];
    float_type ux_array[size];
    float_type vx_array[size];
    float_type hx_array[size];
    float_type uy_array[size];
    float_type vy_array[size];
    float_type hy_array[size];

    data_store_t u;
    data_store_t v;
    data_store_t h;
    data_store_t ux;
    data_store_t vx;
    data_store_t hx;
    data_store_t uy;
    data_store_t vy;
    data_store_t hy;

    static float_type dx() { return 1.; }
    static float_type dy() { return 1.; }
    static float_type dt() { return .02; }
    static float_type g() { return 9.81; }

    static constexpr float_type height = 2.;
    GT_FUNCTION
    static float_type droplet(uint_t const &i, uint_t const &j) {
#ifndef __CUDACC__
        return 1. +
               height * std::exp(-5 * (((i - 3) * dx()) * (((i - 3) * dx())) + ((j - 7) * dy()) * ((j - 7) * dy())));
#else // if CUDA we test the serial case
        return 1. +
               height * std::exp(-5 * (((i - 3) * dx()) * (((i - 3) * dx())) + ((j - 3) * dy()) * ((j - 3) * dy())));
#endif
    }

    shallow_water_reference()
        : solution_meta(DimI, DimJ, static_cast< uint_t >(1)), solution(solution_meta, "solution"),
          u(solution_meta, u_array, enumtype::ExternalCPU, "u"), v(solution_meta, v_array, enumtype::ExternalCPU, "v"),
          h(solution_meta, h_array, enumtype::ExternalCPU, "h"),
          ux(solution_meta, ux_array, enumtype::ExternalCPU, "ux"),
          vx(solution_meta, vx_array, enumtype::ExternalCPU, "vx"),
          hx(solution_meta, hx_array, enumtype::ExternalCPU, "hx"),
          uy(solution_meta, uy_array, enumtype::ExternalCPU, "uy"),
          vy(solution_meta, vy_array, enumtype::ExternalCPU, "vy"),
          hy(solution_meta, hy_array, enumtype::ExternalCPU, "hy") {}

    void setup() {
        for (uint_t i = 0; i < DimI; ++i)
            for (uint_t j = 0; j < DimJ; ++j) {
                uint_t id = i * strides[0] + j * strides[1];
                u.get_storage_ptr()->get_cpu_ptr()[id] = 0;
                v.get_storage_ptr()->get_cpu_ptr()[id] = 0;
                h.get_storage_ptr()->get_cpu_ptr()[id] = droplet(i, j);
                ux.get_storage_ptr()->get_cpu_ptr()[id] = 0;
                vx.get_storage_ptr()->get_cpu_ptr()[id] = 0;
                hx.get_storage_ptr()->get_cpu_ptr()[id] = 0;
                uy.get_storage_ptr()->get_cpu_ptr()[id] = 0;
                vy.get_storage_ptr()->get_cpu_ptr()[id] = 0;
                hy.get_storage_ptr()->get_cpu_ptr()[id] = 0;
            }
        solution.template set< 0, 0 >(h);
        solution.template set< 1, 0 >(u);
        solution.template set< 2, 0 >(v);
    }

    void iterate() {
        // get ptrs directly
        auto up = u.get_storage_ptr()->get_cpu_ptr();
        auto vp = v.get_storage_ptr()->get_cpu_ptr();
        auto hp = h.get_storage_ptr()->get_cpu_ptr();
        auto uxp = ux.get_storage_ptr()->get_cpu_ptr();
        auto hxp = hx.get_storage_ptr()->get_cpu_ptr();
        auto vxp = vx.get_storage_ptr()->get_cpu_ptr();
        auto uyp = uy.get_storage_ptr()->get_cpu_ptr();
        auto hyp = hy.get_storage_ptr()->get_cpu_ptr();
        auto vyp = vy.get_storage_ptr()->get_cpu_ptr();
        // check if we are currently working on device or on host
        for (uint_t i = 0; i < DimI - 1; ++i)
            for (uint_t j = 0; j < DimJ - 2; ++j) {
                uint_t id = i * strides[0] + j * strides[1];
                hxp[id] = (hp[id + ip1 + jp1] + hp[id + jp1]) / 2. -
                          (up[id + ip1 + jp1] - up[id + jp1]) * (dt() / (2 * dx()));

                uxp[id] =
                    (up[id + ip1 + jp1] + up[id + jp1]) / 2. -
                    (((pow< 2 >(up[id + ip1 + jp1])) / hp[id + ip1 + jp1] + pow< 2 >(hp[id + ip1 + jp1]) * g() / 2.) -
                        (pow< 2 >(up[id + jp1]) / hp[id + jp1] + pow< 2 >(hp[id + jp1]) * (g() / 2.))) *
                        (dt() / (2. * dx()));

                vxp[id] = (vp[id + ip1 + jp1] + vp[id + jp1]) / 2. -
                          (up[id + ip1 + jp1] * vp[id + ip1 + jp1] / hp[id + ip1 + jp1] -
                              up[id + jp1] * vp[id + jp1] / hp[id + jp1]) *
                              (dt() / (2 * dx()));
            }

        for (uint_t i = 0; i < DimI - 2; ++i)
            for (uint_t j = 0; j < DimJ - 1; ++j) {
                uint_t id = i * strides[0] + j * strides[1];
                hyp[id] = (hp[id + ip1 + jp1] + hp[id + ip1]) / 2. -
                          (vp[id + ip1 + jp1] - vp[id + ip1]) * (dt() / (2 * dy()));

                uyp[id] = (up[id + ip1 + jp1] + up[id + ip1]) / 2. -
                          (vp[id + ip1 + jp1] * up[id + ip1 + jp1] / hp[id + ip1 + jp1] -
                              vp[id + ip1] * up[id + ip1] / hp[id + ip1]) *
                              (dt() / (2 * dy()));

                vyp[id] =
                    (vp[id + ip1 + jp1] + vp[id + ip1]) / 2. -
                    ((pow< 2 >(vp[id + ip1 + jp1]) / hp[id + ip1 + jp1] + pow< 2 >(hp[id + ip1 + jp1]) * g() / 2.) -
                        (pow< 2 >(vp[id + ip1]) / hp[id + ip1] + pow< 2 >(hp[id + ip1]) * (g() / 2.))) *
                        (dt() / (2. * dy()));
            }

        for (uint_t i = 1; i < DimI - 2; ++i)
            for (uint_t j = 1; j < DimJ - 2; ++j) {
                uint_t id = i * strides[0] + j * strides[1];
                hp[id] = hp[id] - (uxp[id + jm1] - uxp[id + im1 + jm1]) * (dt() / dx()) -
                         (vyp[id + im1] - vyp[id + im1 + jm1]) * (dt() / dy());

                up[id] = up[id] -
                         (pow< 2 >(uxp[id + jm1]) / hxp[id + jm1] + hxp[id + jm1] * hxp[id + jm1] * ((g() / 2.)) -
                             (pow< 2 >(uxp[id + im1 + jm1]) / hxp[id + im1 + jm1] +
                                 pow< 2 >(hxp[id + im1 + jm1]) * ((g() / 2.)))) *
                             ((dt() / dx())) -
                         (vyp[id + im1] * uyp[id + im1] / hyp[id + im1] -
                             vyp[id + im1 + jm1] * uyp[id + im1 + jm1] / hyp[id + im1 + jm1]) *
                             (dt() / dy());

                vp[id] = vp[id] -
                         (uxp[id + jm1] * vxp[id + jm1] / hxp[id + jm1] -
                             (uxp[id + im1 + jm1] * vxp[id + im1 + jm1]) / hxp[id + im1 + jm1]) *
                             ((dt() / dx())) -
                         (pow< 2 >(vyp[id + im1]) / hyp[id + im1] + pow< 2 >(hyp[id + im1]) * ((g() / 2.)) -
                             (pow< 2 >(vyp[id + im1 + jm1]) / hyp[id + im1 + jm1] +
                                 pow< 2 >(hyp[id + im1 + jm1]) * ((g() / 2.)))) *
                             ((dt() / dy()));
            }
    }
};

template < typename StorageType, uint_t DimI, uint_t DimJ >
constexpr uint_t shallow_water_reference< StorageType, DimI, DimJ >::strides[2];
template < typename StorageType, uint_t DimI, uint_t DimJ >
constexpr uint_t shallow_water_reference< StorageType, DimI, DimJ >::size;
template < typename StorageType, uint_t DimI, uint_t DimJ >
constexpr uint_t shallow_water_reference< StorageType, DimI, DimJ >::ip1;
template < typename StorageType, uint_t DimI, uint_t DimJ >
constexpr uint_t shallow_water_reference< StorageType, DimI, DimJ >::jp1;
template < typename StorageType, uint_t DimI, uint_t DimJ >
constexpr uint_t shallow_water_reference< StorageType, DimI, DimJ >::im1;
template < typename StorageType, uint_t DimI, uint_t DimJ >
constexpr uint_t shallow_water_reference< StorageType, DimI, DimJ >::jm1;
