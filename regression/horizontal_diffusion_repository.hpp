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

#include "backend_select.hpp"
#include <gridtools/gridtools.hpp>
#include <gridtools/storage/storage-facility.hpp>

namespace horizontal_diffusion {

    using storage_tr = gridtools::storage_traits<backend_t::backend_id_t>;

    using gridtools::int_t;
    using gridtools::uint_t;

    using storage_info_ijk_t = storage_tr::storage_info_t<0, 3, gridtools::halo<2, 2, 0>>;
    using storage_info_ij_t =
        storage_tr::special_storage_info_t<1, gridtools::selector<1, 1, 0>, gridtools::halo<2, 2, 0>>;
    using storage_info_j_t =
        storage_tr::special_storage_info_t<2, gridtools::selector<0, 1, 0>, gridtools::halo<0, 0, 0>>;
    using storage_info_scalar_t =
        storage_tr::special_storage_info_t<3, gridtools::selector<0, 0, 0>, gridtools::halo<0, 0, 0>>;

    class repository {
      public:
        using storage_type = storage_tr::data_store_t<gridtools::float_type, storage_info_ijk_t>;
        using ij_storage_type = storage_tr::data_store_t<gridtools::float_type, storage_info_ij_t>;
        using j_storage_type = storage_tr::data_store_t<gridtools::float_type, storage_info_j_t>;
        using scalar_storage_type = storage_tr::data_store_t<gridtools::float_type, storage_info_scalar_t>;

        storage_info_ijk_t m_storage_info_ijk;
        storage_info_j_t m_storage_info_j;

      private:
        storage_info_ij_t m_storage_info_ij;
        storage_info_scalar_t m_storage_info_scalar;
        storage_type in_, out_, out_ref_, coeff_;
        j_storage_type crlato_, crlatu_, crlat0_, crlat1_;
        const uint_t halo_size_;
        const uint_t idim_, jdim_, kdim_;

      public:
        repository(const uint_t idim, const uint_t jdim, const uint_t kdim, const uint_t halo_size)
            : m_storage_info_ijk(idim, jdim, kdim), m_storage_info_ij(idim, jdim, kdim), m_storage_info_j(1, jdim, 1),
              m_storage_info_scalar(1, 1, 1), in_(m_storage_info_ijk, "in"), crlato_(m_storage_info_j, "crlato"),
              crlatu_(m_storage_info_j, "crlatu"), crlat0_(m_storage_info_j, "crlat0"),
              crlat1_(m_storage_info_j, "crlat1"), out_(m_storage_info_ijk, "out"),
              out_ref_(m_storage_info_ijk, "out_ref"), coeff_(m_storage_info_ijk, "coeff"), halo_size_(halo_size),
              idim_(idim), jdim_(jdim), kdim_(kdim) {}

        void init_fields() {
            const double PI = std::atan(1.) * 4.;
            init_field_to_value(out_, 0.0);
            init_field_to_value(out_ref_, 0.0);

            auto v_in = make_host_view(in_);
            auto v_coeff = make_host_view(coeff_);
            auto v_crlato = make_host_view(crlato_);
            auto v_crlatu = make_host_view(crlatu_);
            auto v_crlat0 = make_host_view(crlat0_);
            auto v_crlat1 = make_host_view(crlat1_);

            const uint_t i_begin = 0;
            const uint_t i_end = idim_;
            const uint_t j_begin = 0;
            const uint_t j_end = jdim_;
            const uint_t k_begin = 0;
            const uint_t k_end = kdim_;

            double dx = 1. / (double)(i_end - i_begin);
            double dy = 1. / (double)(j_end - j_begin);

            double delta0 = (0.995156 - 0.994954) / (double)(jdim_ - 1);
            double delta1 = (0.995143 - 0.994924) / (double)(jdim_ - 1);

            for (uint_t j = j_begin; j < j_end; j++) {
                v_crlat0(0, j, 0) = 0.994954 + (double)(j)*delta0;
                v_crlat1(0, j, 0) = 0.994924 + (double)(j)*delta1;

                for (uint_t i = i_begin; i < i_end; i++) {
                    double x = dx * (double)(i - i_begin);
                    double y = dy * (double)(j - j_begin);
                    for (uint_t k = k_begin; k < k_end; k++) {
                        // in values between 5 and 9
                        v_in(i, j, k) = 5. + 8 * (2. + cos(PI * (x + 1.5 * y)) + sin(2 * PI * (x + 1.5 * y))) / 4.;

                        // coefficient values
                        v_coeff(i, j, k) = 0.025;
                    }
                }
            }
            for (uint_t j = j_begin + 1; j < j_end; j++) {
                v_crlato(0, j, 0) = v_crlat1(0, j, 0) / v_crlat0(0, j, 0);
                v_crlatu(0, j, 0) = v_crlat1(0, j - 1, 0) / v_crlat0(0, j, 0);
            }
            v_crlato(0, j_begin, 0) = v_crlat1(0, j_begin, 0) / v_crlat0(0, j_begin, 0);
        }

        template <typename TStorage_type, typename TValue_type>
        void init_field_to_value(TStorage_type field, TValue_type value) {
            const uint_t dim0 = (TStorage_type::storage_info_t::layout_t::template at<0>() == -1) ? 1 : idim_;
            const uint_t dim1 = (TStorage_type::storage_info_t::layout_t::template at<1>() == -1) ? 1 : jdim_;
            const uint_t dim2 = (TStorage_type::storage_info_t::layout_t::template at<2>() == -1) ? 1 : kdim_;

            auto v = make_host_view(field);
            for (uint_t k = 0; k < dim2; ++k) {
                for (uint_t i = 0; i < dim0; ++i) {
                    for (uint_t j = 0; j < dim1; ++j) {
                        v(i, j, k) = value;
                    }
                }
            }
        }

        void generate_reference() {
            ij_storage_type lap(m_storage_info_ij, "lap");
            ij_storage_type flx(m_storage_info_ij, "flx");
            ij_storage_type fly(m_storage_info_ij, "fly");

            init_field_to_value(lap, 0.0);

            auto v_in = make_host_view(in_);
            auto v_lap = make_host_view(lap);
            auto v_flx = make_host_view(flx);
            auto v_fly = make_host_view(fly);

            auto v_out = make_host_view(out());
            auto v_out_ref = make_host_view(out_ref_);
            auto v_coeff = make_host_view(coeff_);

            // kbody
            for (uint_t k = 0; k < kdim_; ++k) {
                for (uint_t i = halo_size_ - 1; i < idim_ - halo_size_ + 1; ++i) {
                    for (uint_t j = halo_size_ - 1; j < jdim_ - halo_size_ + 1; ++j) {
                        v_lap(i, j, (uint_t)0) =
                            (gridtools::float_type)4 * v_in(i, j, k) -
                            (v_in(i + 1, j, k) + v_in(i, j + 1, k) + v_in(i - 1, j, k) + v_in(i, j - 1, k));
                    }
                }
                for (uint_t i = halo_size_ - 1; i < idim_ - halo_size_; ++i) {
                    for (uint_t j = halo_size_; j < jdim_ - halo_size_; ++j) {
                        v_flx(i, j, (uint_t)0) = v_lap(i + 1, j, (uint_t)0) - v_lap(i, j, (uint_t)0);
                        if (v_flx(i, j, (uint_t)0) * (v_in(i + 1, j, k) - v_in(i, j, k)) > 0)
                            v_flx(i, j, (uint_t)0) = 0.;
                    }
                }
                for (uint_t i = halo_size_; i < idim_ - halo_size_; ++i) {
                    for (uint_t j = halo_size_ - 1; j < jdim_ - halo_size_; ++j) {
                        v_fly(i, j, (uint_t)0) = v_lap(i, j + 1, (uint_t)0) - v_lap(i, j, (uint_t)0);
                        if (v_fly(i, j, (uint_t)0) * (v_in(i, j + 1, k) - v_in(i, j, k)) > 0)
                            v_fly(i, j, (uint_t)0) = 0.;
                    }
                }
                for (uint_t i = halo_size_; i < idim_ - halo_size_; ++i) {
                    for (uint_t j = halo_size_; j < jdim_ - halo_size_; ++j) {
                        v_out_ref(i, j, k) =
                            v_in(i, j, k) - v_coeff(i, j, k) * (v_flx(i, j, (uint_t)0) - v_flx(i - 1, j, (uint_t)0) +
                                                                   v_fly(i, j, (uint_t)0) - v_fly(i, j - 1, (uint_t)0));
                    }
                }
            }
        }

        void generate_reference_simple() {
            ij_storage_type lap(m_storage_info_ij, "lap");

            init_field_to_value(lap, 0.0);

            auto v_in = make_host_view(in_);
            auto v_lap = make_host_view(lap);
            auto v_crlato = make_host_view(crlato_);
            auto v_crlatu = make_host_view(crlatu_);

            auto v_out_ref = make_host_view(out_ref_);
            auto v_coeff = make_host_view(coeff_);

            // kbody
            for (uint_t k = 0; k < kdim_; ++k) {
                for (uint_t i = halo_size_ - 1; i < idim_ - halo_size_ + 1; ++i) {
                    for (uint_t j = halo_size_ - 1; j < jdim_ - halo_size_ + 1; ++j) {
                        v_lap(i, j, (uint_t)0) = v_in(i + 1, j, k) + v_in(i - 1, j, k) -
                                                 (gridtools::float_type)2 * v_in(i, j, k) +
                                                 v_crlato(i, j, k) * (v_in(i, j + 1, k) - v_in(i, j, k)) +
                                                 v_crlatu(i, j, k) * (v_in(i, j - 1, k) - v_in(i, j, k));
                    }
                }
                for (uint_t i = halo_size_; i < idim_ - halo_size_; ++i) {
                    for (uint_t j = halo_size_; j < jdim_ - halo_size_; ++j) {
                        gridtools::float_type fluxx = v_lap(i + 1, j, k) - v_lap(i, j, k);
                        gridtools::float_type fluxx_m = v_lap(i, j, k) - v_lap(i - 1, j, k);

                        gridtools::float_type fluxy = v_crlato(i, j, k) * (v_lap(i, j + 1, k) - v_lap(i, j, k));
                        gridtools::float_type fluxy_m = v_crlato(i, j, k) * (v_lap(i, j, k) - v_lap(i, j - 1, k));

                        v_out_ref(i, j, k) = v_in(i, j, k) + ((fluxx_m - fluxx) + (fluxy_m - fluxy)) * v_coeff(i, j, k);
                    }
                }
            }
        }

        storage_type &in() { return in_; }
        storage_type &out() { return out_; }
        storage_type &out_ref() { return out_ref_; }
        storage_type &coeff() { return coeff_; }
        j_storage_type &crlato() { return crlato_; }
        j_storage_type &crlatu() { return crlatu_; }

        storage_info_ijk_t &storage_info_ijk() { return m_storage_info_ijk; }
    };
} // namespace horizontal_diffusion
