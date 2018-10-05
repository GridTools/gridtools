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

#include <gridtools/common/defs.hpp>

namespace gridtools {
    class horizontal_diffusion_repository {
        using fun_t = std::function<double(int_t, int_t, int_t)>;
        using j_fun_t = std::function<double(int_t)>;
        uint_t idim_, jdim_, kdim_;

        double delta0_ = (0.995156 - 0.994954) / (jdim_ - 1.);
        double delta1_ = (0.995143 - 0.994924) / (jdim_ - 1.);

        j_fun_t crlat0_ = [this](int_t j) { return 0.994954 + j * delta0_; };
        j_fun_t crlat1_ = [this](int_t j) { return 0.994924 + j * delta1_; };

      public:
        horizontal_diffusion_repository(uint_t idim, uint_t jdim, uint_t kdim)
            : idim_(idim), jdim_(jdim), kdim_(kdim) {}

        fun_t in = [this](int_t i, int_t j, int_t) {
            const double PI = std::atan(1.) * 4.;
            double dx = 1. / idim_;
            double dy = 1. / jdim_;
            double x = dx * i;
            double y = dy * j;
            // in values between 5 and 9
            return 5. + 8 * (2. + cos(PI * (x + 1.5 * y)) + sin(2 * PI * (x + 1.5 * y))) / 4.;
        };

        fun_t coeff = [](int_t, int_t, int_t) { return 0.025; };

        fun_t crlato = [this](int_t, int_t j, int_t) { return crlat1_(j) / crlat0_(j); };

        fun_t crlatu = [this](int_t, int_t j, int_t) { return j == 0 ? 0 : crlat1_(j - 1) / crlat0_(j); };

        fun_t out = [this](int_t i, int_t j, int_t k) {
            auto lap = [this, k](int_t ii, int_t jj) {
                return 4 * in(ii, jj, k) -
                       (in(ii + 1, jj, k) + in(ii, jj + 1, k) + in(ii - 1, jj, k) + in(ii, jj - 1, k));
            };
            auto flx = [this, k, lap](int_t ii, int_t jj) {
                double res = lap(ii + 1, jj) - lap(ii, jj);
                if (res * (in(ii + 1, jj, k) - in(ii, jj, k)) > 0)
                    res = 0.;
                return res;
            };
            auto fly = [this, k, lap](int_t ii, int_t jj) {
                auto res = lap(ii, jj + 1) - lap(ii, jj);
                if (res * (in(ii, jj + 1, k) - in(ii, jj, k)) > 0)
                    res = 0.;
                return res;
            };
            return in(i, j, k) - coeff(i, j, k) * (flx(i, j) - flx(i - 1, j) + fly(i, j) - fly(i, j - 1));
        };

        fun_t out_simple = [this](int_t i, int_t j, int_t k) {
            auto lap = [this, k](int_t ii, int_t jj) {
                return in(ii + 1, jj, k) + in(ii - 1, jj, k) - 2 * in(ii, jj, k) +
                       crlato(ii, jj, k) * (in(ii, jj + 1, k) - in(ii, jj, k)) +
                       crlatu(ii, jj, k) * (in(ii, jj - 1, k) - in(ii, jj, k));
            };
            auto fluxx = lap(i + 1, j) - lap(i, j);
            auto fluxx_m = lap(i, j) - lap(i - 1, j);
            auto fluxy = crlato(i, j, k) * (lap(i, j + 1) - lap(i, j));
            auto fluxy_m = crlato(i, j, k) * (lap(i, j) - lap(i, j - 1));
            return in(i, j, k) + ((fluxx_m - fluxx) + (fluxy_m - fluxy)) * coeff(i, j, k);
        };
    };
} // namespace gridtools
