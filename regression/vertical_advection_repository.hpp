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

#include <cmath>
#include <functional>

#include "vertical_advection_defs.hpp"
#include <gridtools/common/defs.hpp>

namespace gridtools {

    class vertical_advection_repository {
        using fun_t = std::function<double(int_t, int_t, int_t)>;

        uint_t idim_, jdim_, kdim_;

        double PI = 4 * std::atan(1);

        double utens_stage_ref(int_t i, int_t j, int_t k) {
            double x = 1. * i / idim_;
            double y = 1. * j / jdim_;
            return 7 + 1.25 * (2. + cos(PI * (x + y)) + sin(2 * PI * (x + y))) + .1 * k;
        };

        double ccol(int_t i, int_t j, int_t k) {
            assert(k < kdim_ - 1);
            if (k == 0) {
                double tmp = .25 * BET_P * (wcon(i + 1, j, 1) + wcon(i, j, 1));
                return tmp / (dtr_stage - tmp);
            }
            double gav = -0.25 * (wcon(i + 1, j, k) + wcon(i, j, k));
            double gcv = 0.25 * (wcon(i + 1, j, k + 1) + wcon(i, j, k + 1));
            double tmp = gcv * BET_P;
            return tmp / (dtr_stage - tmp - gav * BET_P * (1 + ccol(i, j, k - 1)));
        };

        double dcol(int_t i, int_t j, int_t k) {
            if (k == 0) {
                double gcv = .25 * (wcon(i + 1, j, 1) + wcon(i, j, 1));
                double correctionTerm = -gcv * BET_M * (u_stage(i, j, 1) - u_stage(i, j, 0));
                return (dtr_stage * u_pos(i, j, 0) + utens(i, j, 0) + utens_stage_ref(i, j, 0) + correctionTerm) /
                       (dtr_stage - gcv * BET_P);
            }
            double gav = -0.25 * (wcon(i + 1, j, k) + wcon(i, j, k));
            double as = gav * BET_M;
            double acol = gav * BET_P;
            double bcol;
            double correctionTerm;
            if (k == kdim_ - 1) {
                bcol = dtr_stage - acol;
                correctionTerm = -as * (u_stage(i, j, k - 1) - u_stage(i, j, k));
            } else {
                double gcv = 0.25 * (wcon(i + 1, j, k + 1) + wcon(i, j, k + 1));
                bcol = dtr_stage - acol - gcv * BET_P;
                correctionTerm = -as * (u_stage(i, j, k - 1) - u_stage(i, j, k)) -
                                 gcv * BET_M * (u_stage(i, j, k + 1) - u_stage(i, j, k));
            }
            return (dtr_stage * u_pos(i, j, k) + utens(i, j, k) + utens_stage_ref(i, j, k) + correctionTerm -
                       dcol(i, j, k - 1) * acol) /
                   (bcol - ccol(i, j, k - 1) * acol);
        }

        double datacol(int_t i, int_t j, int_t k) {
            return k == kdim_ - 1 ? dcol(i, j, k) : dcol(i, j, k) - ccol(i, j, k) * datacol(i, j, k + 1);
        }

      public:
        double dtr_stage = 3. / 20.;

        fun_t u_stage = [this](int_t i, int_t j, int_t) {
            double x = 1. * i / idim_;
            double y = 1. * j / jdim_;
            // u values between 5 and 9
            return 7 + std::cos(PI * (x + y)) + std::sin(2 * PI * (x + y));
        };

        fun_t u_pos = u_stage;

        fun_t wcon = [this](int_t i, int_t j, int_t k) {
            double x = 1. * i / idim_;
            double y = 1. * j / jdim_;
            double z = 1. * k / kdim_;

            // wcon values between -2e-4 and 2e-4 (with zero at k=0)
            return 2e-4 * (-1.07 + (2 + cos(PI * (x + z)) + cos(PI * y)) / 2);
        };

        fun_t utens = [this](int_t i, int_t j, int_t k) {
            double x = 1. * i / idim_;
            double y = 1. * j / jdim_;
            double z = 1. * k / kdim_;

            // utens values between -3e-6 and 3e-6 (with zero at k=0)
            return 3e-6 * (-1.0235 + (2. + cos(PI * (x + y)) + cos(PI * y * z)) / 2);
        };

        fun_t utens_stage = [this](
                                int_t i, int_t j, int_t k) { return dtr_stage * (datacol(i, j, k) - u_pos(i, j, k)); };

        vertical_advection_repository(uint_t idim, uint_t jdim, uint_t kdim) : idim_(idim), jdim_(jdim), kdim_(kdim) {}
    };
} // namespace gridtools
