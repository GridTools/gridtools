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

#include "vertical_advection_defs.hpp"
#include <gridtools.hpp>
#include <storage-facility.hpp>

using gridtools::uint_t;
using gridtools::int_t;

namespace vertical_advection {

    class repository {
      public:
        using storage_info_ijk_t = storage_tr::storage_info_t< 0, 3, gridtools::halo< 3, 3, 0 > >;
        using storage_info_ij_t =
            storage_tr::special_storage_info_t< 1, gridtools::selector< 1, 1, 0 >, gridtools::halo< 3, 3, 0 > >;
        using storage_info_scalar_t =
            storage_tr::special_storage_info_t< 2, gridtools::selector< 0, 0, 0 >, gridtools::halo< 0, 0, 0 > >;

        using storage_type = storage_tr::data_store_t< gridtools::float_type, storage_info_ijk_t >;
        using ij_storage_type = storage_tr::data_store_t< gridtools::float_type, storage_info_ij_t >;
        using scalar_storage_type = storage_tr::data_store_t< gridtools::float_type, storage_info_scalar_t >;

        repository(const uint_t idim, const uint_t jdim, const uint_t kdim, const uint_t halo_size)
            : m_storage_info(idim - (2 * halo_size), jdim - (2 * halo_size), kdim),
              m_scalar_storage_info(1, 1, 1), // fake 3D
              utens_stage_(m_storage_info), utens_stage_ref_(m_storage_info), u_stage_(m_storage_info),
              wcon_(m_storage_info), u_pos_(m_storage_info), utens_(m_storage_info), ipos_(m_storage_info),
              jpos_(m_storage_info), kpos_(m_storage_info),
              // dtr_stage_(0,0,0, -1, "dtr_stage"),
              dtr_stage_(m_scalar_storage_info), halo_size_(halo_size), idim_(idim), jdim_(jdim), kdim_(kdim) {
            init_field_to_value(utens_stage_, -1.);
            init_field_to_value(utens_stage_ref_, -1.);
            init_field_to_value(u_stage_, -1.);
            init_field_to_value(wcon_, -1.);
            init_field_to_value(u_pos_, -1.);
            init_field_to_value(utens_, -1.);
            init_field_to_value(ipos_, -1.);
            init_field_to_value(jpos_, -1.);
            init_field_to_value(kpos_, -1.);
            init_field_to_value(dtr_stage_, -1.);
        }

        void init_fields() {
            // set the fields to advect
            const double PI = std::atan(1.) * 4.;

            const uint_t i_begin = 0;
            const uint_t i_end = idim_;
            const uint_t j_begin = 0;
            const uint_t j_end = jdim_;
            const uint_t k_begin = 0;
            const uint_t k_end = kdim_;

            auto dtr_stage_v = make_host_view(dtr_stage_);
            auto u_stage_v = make_host_view(u_stage_);
            auto u_pos_v = make_host_view(u_pos_);
            auto utens_v = make_host_view(utens_);
            auto wcon_v = make_host_view(wcon_);
            auto utens_stage_v = make_host_view(utens_stage_);
            auto utens_stage_ref_v = make_host_view(utens_stage_ref_);
            auto ipos_v = make_host_view(ipos_);
            auto jpos_v = make_host_view(jpos_);
            auto kpos_v = make_host_view(kpos_);

            double dtadv = (double)20. / 3.;
            dtr_stage_v(0, 0, 0) = (double)1.0 / dtadv;

            double dx = 1. / (double)(i_end - i_begin);
            double dy = 1. / (double)(j_end - j_begin);
            double dz = 1. / (double)(k_end - k_begin);

            for (int j = j_begin; j < j_end; j++) {
                for (int i = i_begin; i < i_end; i++) {
                    double x = dx * (double)(i - i_begin);
                    double y = dy * (double)(j - j_begin);
                    for (int k = k_begin; k < k_end; k++) {
                        double z = dz * (double)(k - k_begin);
                        dtr_stage_v(i, j, k) = (double)1.0 / dtadv;

                        // u values between 5 and 9
                        u_stage_v(i, j, k) = 5. + 4 * (2. + cos(PI * (x + y)) + sin(2 * PI * (x + y))) / 4.;
                        u_pos_v(i, j, k) = 5. + 4 * (2. + cos(PI * (x + y)) + sin(2 * PI * (x + y))) / 4.;
                        // utens values between -3e-6 and 3e-6 (with zero at k=0)
                        utens_v(i, j, k) = 3e-6 * (-1 + 2 * (2. + cos(PI * (x + y)) + cos(PI * y * z)) / 4. +
                                                      0.05 * (0.5 - 24.0) / 50.);
                        // wcon values between -2e-4 and 2e-4 (with zero at k=0)
                        wcon_v(i, j, k) =
                            2e-4 * (-1 + 2 * (2. + cos(PI * (x + z)) + cos(PI * y)) / 4. + 0.1 * (0.5 - 35.5) / 50.);

                        utens_stage_v(i, j, k) =
                            7. + 5 * (2. + cos(PI * (x + y)) + sin(2 * PI * (x + y))) / 4. + k * 0.1;
                        utens_stage_ref_v(i, j, k) = utens_stage_v(i, j, k);
                        ipos_v(i, j, k) = i;
                        jpos_v(i, j, k) = j;
                        kpos_v(i, j, k) = k;
                    }
                }
            }
        }

        template < typename TStorage_type, typename TValue_type >
        void init_field_to_value(TStorage_type &field, TValue_type value) {
            auto field_v = make_host_view(field);
            const uint_t dim0 = (TStorage_type::storage_info_t::layout_t::template at< 0 >() == -1) ? 1 : idim_;
            const uint_t dim1 = (TStorage_type::storage_info_t::layout_t::template at< 1 >() == -1) ? 1 : jdim_;
            const uint_t dim2 = (TStorage_type::storage_info_t::layout_t::template at< 2 >() == -1) ? 1 : kdim_;

            for (uint_t k = 0; k < dim2; ++k) {
                for (uint_t i = 0; i < dim0; ++i) {
                    for (uint_t j = 0; j < dim1; ++j) {
                        field_v(i, j, k) = value;
                    }
                }
            }
        }

        void generate_reference() {
            auto dtr_stage_v = make_host_view(dtr_stage_);
            double dtr_stage = dtr_stage_v(0, 0, 0);

            storage_info_ij_t storage_info_ij(idim_ - (2 * halo_size_), jdim_ - (2 * halo_size_), (uint_t)1);
            ij_storage_type datacol(storage_info_ij);
            storage_info_ijk_t storage_info_(idim_ - (2 * halo_size_), jdim_ - (2 * halo_size_), kdim_);
            storage_type ccol(storage_info_);
            storage_type dcol(storage_info_);

            init_field_to_value(ccol, 0.0);
            init_field_to_value(dcol, 0.0);

            init_field_to_value(datacol, 0.0);

            // Generate U
            forward_sweep(1, 0, ccol, dcol);
            backward_sweep(ccol, dcol, datacol);
        }

        void forward_sweep(int ishift, int jshift, storage_type &ccol, storage_type &dcol) {
            auto dtr_stage_v = make_host_view(dtr_stage_);
            auto wcon_v = make_host_view(wcon_);
            auto ccol_v = make_host_view(ccol);
            auto dcol_v = make_host_view(dcol);
            auto u_stage_v = make_host_view(u_stage_);
            auto utens_stage_ref_v = make_host_view(utens_stage_ref_);
            auto utens_v = make_host_view(utens_);
            auto u_pos_v = make_host_view(u_pos_);

            double dtr_stage = dtr_stage_v(0, 0, 0);
            // k minimum
            int k = 0;
            for (int i = halo_size_; i < idim_ - halo_size_; ++i) {
                for (int j = halo_size_; j < jdim_ - halo_size_; ++j) {
                    double gcv = (double)0.25 * (wcon_v(i + ishift, j + jshift, k + 1) + wcon_v(i, j, k + 1));
                    double cs = gcv * BET_M;

                    ccol_v(i, j, k) = gcv * BET_P;
                    double bcol = dtr_stage_v(0, 0, 0) - ccol_v(i, j, k);

                    // update the d column
                    double correctionTerm = -cs * (u_stage_v(i, j, k + 1) - u_stage_v(i, j, k));
                    dcol_v(i, j, k) =
                        dtr_stage * u_pos_v(i, j, k) + utens_v(i, j, k) + utens_stage_ref_v(i, j, k) + correctionTerm;

                    double divided = (double)1.0 / bcol;
                    ccol_v(i, j, k) = ccol_v(i, j, k) * divided;
                    dcol_v(i, j, k) = dcol_v(i, j, k) * divided;

                    // if(i==3 && j == 3)
                    // std::cout << "AT ref at  " << k << "  " << bcol << "  " << ccol(i,j,k) << " " << dcol(i,j,k) << "
                    // " << gcv <<
                    // "  " << wcon_(i,j,k+1) << "  " << wcon_(i+ishift, j+jshift, k+1) << std::endl;
                }
            }

            // kbody
            for (k = 1; k < kdim_ - 1; ++k) {
                for (int i = halo_size_; i < idim_ - halo_size_; ++i) {
                    for (int j = halo_size_; j < jdim_ - halo_size_; ++j) {
                        double gav = (double)-0.25 * (wcon_v(i + ishift, j + jshift, k) + wcon_v(i, j, k));
                        double gcv = (double)0.25 * (wcon_v(i + ishift, j + jshift, k + 1) + wcon_v(i, j, k + 1));

                        double as = gav * BET_M;
                        double cs = gcv * BET_M;

                        double acol = gav * BET_P;
                        ccol_v(i, j, k) = gcv * BET_P;
                        double bcol = dtr_stage - acol - ccol_v(i, j, k);

                        double correctionTerm = -as * (u_stage_v(i, j, k - 1) - u_stage_v(i, j, k)) -
                                                cs * (u_stage_v(i, j, k + 1) - u_stage_v(i, j, k));
                        dcol_v(i, j, k) = dtr_stage * u_pos_v(i, j, k) + utens_v(i, j, k) + utens_stage_ref_v(i, j, k) +
                                          correctionTerm;

                        double divided = (double)1.0 / (bcol - (ccol_v(i, j, k - 1) * acol));
                        ccol_v(i, j, k) = ccol_v(i, j, k) * divided;
                        dcol_v(i, j, k) = (dcol_v(i, j, k) - (dcol_v(i, j, k - 1) * acol)) * divided;
                        // if(i==3 && j == 3)
                        // std::cout << "FORDW REF at  " << k << "  " << acol << "  " << bcol << "  " << ccol(i,j,k) <<
                        // " " << dcol(i,j,k) << std::endl;
                    }
                }
            }

            // k maximum
            k = kdim_ - 1;
            for (int i = halo_size_; i < idim_ - halo_size_; ++i) {
                for (int j = halo_size_; j < jdim_ - halo_size_; ++j) {
                    double gav = -(double)0.25 * (wcon_v(i + ishift, j + jshift, k) + wcon_v(i, j, k));
                    double as = gav * BET_M;

                    double acol = gav * BET_P;
                    double bcol = dtr_stage - acol;

                    // update the d column
                    double correctionTerm = -as * (u_stage_v(i, j, k - 1) - u_stage_v(i, j, k));
                    dcol_v(i, j, k) =
                        dtr_stage * u_pos_v(i, j, k) + utens_v(i, j, k) + utens_stage_ref_v(i, j, k) + correctionTerm;

                    double divided = (double)1.0 / (bcol - (ccol_v(i, j, k - 1) * acol));
                    dcol_v(i, j, k) = (dcol_v(i, j, k) - (dcol_v(i, j, k - 1) * acol)) * divided;
                }
            }
        }

        void backward_sweep(storage_type &ccol, storage_type &dcol, ij_storage_type &datacol) {
            auto dtr_stage_v = make_host_view(dtr_stage_);
            auto ccol_v = make_host_view(ccol);
            auto dcol_v = make_host_view(dcol);
            auto datacol_v = make_host_view(datacol);
            auto utens_stage_ref_v = make_host_view(utens_stage_ref_);
            auto u_pos_v = make_host_view(u_pos_);

            double dtr_stage = dtr_stage_v(0, 0, 0);
            // k maximum
            int k = kdim_ - 1;
            for (int i = halo_size_; i < idim_ - halo_size_; ++i) {
                for (int j = halo_size_; j < jdim_ - halo_size_; ++j) {
                    datacol_v(i, j, k) = dcol_v(i, j, k);
                    ccol_v(i, j, k) = datacol_v(i, j, k);
                    utens_stage_ref_v(i, j, k) = dtr_stage * (datacol_v(i, j, k) - u_pos_v(i, j, k));
                }
            }
            // kbody
            for (k = kdim_ - 2; k >= 0; --k) {
                for (int i = halo_size_; i < idim_ - halo_size_; ++i) {
                    for (int j = halo_size_; j < jdim_ - halo_size_; ++j) {
                        datacol_v(i, j, k) = dcol_v(i, j, k) - (ccol_v(i, j, k) * datacol_v(i, j, k + 1));
                        ccol_v(i, j, k) = datacol_v(i, j, k);
                        utens_stage_ref_v(i, j, k) = dtr_stage * (datacol_v(i, j, k) - u_pos_v(i, j, k));
                    }
                }
            }
        }

        storage_type &utens_stage() { return utens_stage_; }
        storage_type &wcon() { return wcon_; }
        storage_type &u_pos() { return u_pos_; }
        storage_type &utens() { return utens_; }
        storage_type &ipos() { return ipos_; }
        storage_type &jpos() { return jpos_; }
        storage_type &kpos() { return kpos_; }
        scalar_storage_type &dtr_stage() { return dtr_stage_; }

        // output fields
        storage_type &u_stage() { return u_stage_; }
        storage_type &utens_stage_ref() { return utens_stage_ref_; }

      private:
        storage_info_ijk_t m_storage_info;
        storage_info_scalar_t m_scalar_storage_info;
        storage_type utens_stage_, u_stage_, wcon_, u_pos_, utens_, utens_stage_ref_;
        storage_type ipos_, jpos_, kpos_;
        scalar_storage_type dtr_stage_;
        const uint_t halo_size_;
        const uint_t idim_, jdim_, kdim_;
    };
}
