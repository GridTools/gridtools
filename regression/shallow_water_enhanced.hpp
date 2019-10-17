/*
 * GridTools
 *
 * Copyright (c) 2014-2019, ETH Zurich
 * All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */
#pragma once

// [includes]
#include <fstream>
#include <iostream>
#include <sstream>

#include <gridtools/communication/halo_exchange.hpp>
#include <gridtools/stencil_composition/stencil_composition.hpp>
#include <gridtools/storage/storage_facility.hpp>
#include <gridtools/tools/backend_select.hpp>
#include <gridtools/tools/verifier.hpp>

// [includes]

/*
  @file
  @brief This file shows an implementation of the "shallow water" stencil, with periodic boundary conditions

  For an exhaustive description of the shallow water problem refer to:
  http://www.mathworks.ch/moler/exm/chapters/water.pdf

  NOTE: It is the most human readable and efficient solution among the versions implemented, but it must be compiled for
  the host, with Clang or GCC>=4.9, and with C++11 enabled
*/

// [namespaces]
using namespace gridtools;
using namespace expressions;
// [namespaces]

template <typename DX, typename DY, typename H>
GT_FUNCTION float_type droplet_(uint_t i, uint_t j, DX dx, DY dy, H height) {
#ifndef __CUDACC__
    return 1. + height * std::exp(-5 * (((i - 3) * dx) * (((i - 3) * dx)) + ((j - 7) * dy) * ((j - 7) * dy)));
#else // if CUDA we test the serial case
    return 1. + height * std::exp(-5 * (((i - 3) * dx) * (((i - 3) * dx)) + ((j - 3) * dy) * ((j - 3) * dy)));
#endif
}

#include "shallow_water_reference.hpp"

namespace shallow_water {
    // [functor_traits]
    /**@brief This traits class defined the necessary typesand functions used by all the functors defining the shallow
     * water model*/
    struct functor_traits {

        /**@brief space discretization step in direction i */
        GT_FUNCTION
        static float_type dx() { return 1.; }
        /**@brief space discretization step in direction j */
        GT_FUNCTION
        static float_type dy() { return 1.; }
        /**@brief time discretization step */
        GT_FUNCTION
        static float_type dt() { return .02; }
        /**@brief gravity acceleration */
        GT_FUNCTION
        static float_type g() { return 9.81; }
    };

    template <uint_t Component = 0, uint_t Snapshot = 0>
    struct bc_periodic : functor_traits {
        //! [droplet]
        static constexpr float_type height = 2.;
        GT_FUNCTION
        static float_type droplet(uint_t i, uint_t j) { return droplet_(i, j, dx(), dy(), height); }
        //! [droplet]
    };
    // [boundary_conditions]

    // These are the stencil operators that compose the multistage stencil in this test
    // [flux_x]
    struct flux_x : public functor_traits {

        /** (input) is the solution at the cell center, computed at the previous time level */
        //! [accessor]
        using h = accessor<3, intent::in, extent<0, -1, 0, 0>>;
        //! [accessor]
        using u = accessor<4, intent::in, extent<0, -1, 0, 0>>;
        using v = accessor<5, intent::in, extent<0, -1, 0, 0>>;

        /** (output) is the flux computed on the left edge of the cell */
        using hx = accessor<0, intent::inout>;
        using ux = accessor<1, intent::inout>;
        using vx = accessor<2, intent::inout>;

        using param_list = make_param_list<hx, ux, vx, h, u, v>;

        template <typename Evaluation>
        GT_FUNCTION static void apply(Evaluation &eval) {

            const float_type &tl = 2.;
            dimension<1> i;

            //! [expression]
            eval(hx()) = eval((h() + h(i - 1)) / tl - (u() - u(i - 1)) * (dt() / (2 * dx())));
            //! [expression]

            eval(ux()) =
                eval((u() + u(i - 1)) / tl - ((pow<2>(u()) / h() + pow<2>(h()) * g() / tl) -
                                                 (pow<2>(u(i - 1)) / h(i - 1) + pow<2>(h(i - 1)) * (g() / tl))) *
                                                 (dt() / (tl * dx())));

            eval(vx()) =
                eval((v() + v(i - 1)) / tl - (u() * v() / h() - u(i - 1) * v(i - 1) / h(i - 1)) * (dt() / (2 * dx())));
        }
    };
    // [flux_x]

    // [flux_y]
    struct flux_y : public functor_traits {

        /** (output) is the flux at the bottom edge of the cell */
        using hy = accessor<0, intent::inout>;
        using uy = accessor<1, intent::inout>;
        using vy = accessor<2, intent::inout>;

        /** (input) is the solution at the cell center, computed at the previous time level */
        using h = accessor<3, intent::in, extent<0, 0, 0, -1>>;
        using u = accessor<4, intent::in, extent<0, 0, 0, -1>>;
        using v = accessor<5, intent::in, extent<0, 0, 0, -1>>;

        using param_list = make_param_list<hy, uy, vy, h, u, v>;

        template <typename Evaluation>
        GT_FUNCTION static void apply(Evaluation &eval) {

            const float_type &tl = 2.;
            dimension<2> j;

            eval(hy()) = eval((h() + h(j - 1)) / tl - (v() - v(j - 1)) * (dt() / (2 * dy())));

            eval(uy()) =
                eval((u() + u(j - 1)) / tl - (v() * u() / h() - v(j - 1) * u(j - 1) / h(j - 1)) * (dt() / (2 * dy())));

            eval(vy()) =
                eval((v() + v(j - 1)) / tl - ((pow<2>(v()) / h() + pow<2>(h()) * g() / tl) -
                                                 (pow<2>(v(j - 1)) / h(j - 1) + pow<2>(h(j - 1)) * (g() / tl))) *
                                                 (dt() / (tl * dy())));
        }
    };
    // [flux_y]

    // [final_step]
    struct final_step : public functor_traits {

        /** (input) is the flux at the left edge of the cell */
        using hx = accessor<0, intent::in, extent<0, 1, 0, 1>>;
        using ux = accessor<1, intent::in, extent<0, 1, 0, 1>>;
        using vx = accessor<2, intent::in, extent<0, 1, 0, 1>>;

        /** (input) is the flux at the bottom edge of the cell */
        using hy = accessor<3, intent::in, extent<0, 1, 0, 1>>;
        using uy = accessor<4, intent::in, extent<0, 1, 0, 1>>;
        using vy = accessor<5, intent::in, extent<0, 1, 0, 1>>;

        /** (output) is the solution at the cell center, computed at the previous time level */
        using h = accessor<6, intent::inout>;
        using u = accessor<7, intent::inout>;
        using v = accessor<8, intent::inout>;

        using param_list = make_param_list<hx, ux, vx, hy, uy, vy, h, u, v>;
        static uint_t current_time;

        //########## FINAL STEP #############
        // data dependencies with the previous parts
        // notation: alias<tmp, comp, step>::set<0, 0>() is ==> tmp(comp(0), step(0)).
        // Using a strategy to define some arguments beforehand

        template <typename Evaluation>
        GT_FUNCTION static void apply(Evaluation &eval) {
            const float_type &tl = 2.;
            dimension<1> i;
            dimension<2> j;

            eval(h()) = eval(h() - (ux(i + 1) - ux()) * (dt() / dx()) - (vy(j + 1) - vy()) * (dt() / dy()));

            eval(u()) = eval(u() -
                             (pow<2>(ux(i + 1)) / hx(i + 1) + hx(i + 1) * hx(i + 1) * ((g() / tl)) -
                                 (pow<2>(ux()) / hx() + pow<2>(hx()) * ((g() / tl)))) *
                                 ((dt() / dx())) -
                             (vy(j + 1) * uy(j + 1) / hy(j + 1) - vy() * uy() / hy()) * (dt() / dy()));

            eval(v()) = eval(v() - (ux(i + 1) * vx(i + 1) / hx(i + 1) - (ux() * vx()) / hx()) * ((dt() / dx())) -
                             (pow<2>(vy(j + 1)) / hy(j + 1) + pow<2>(hy(j + 1)) * ((g() / tl)) -
                                 (pow<2>(vy()) / hy() + pow<2>(hy()) * ((g() / tl)))) *
                                 ((dt() / dy())));
        }
    };
    // [final_step]

    uint_t final_step::current_time = 0;

    /*
     * The following operators and structs are for debugging only
     */
    std::ostream &operator<<(std::ostream &s, flux_x const) {
        return s << "initial step 1: ";
        // initiali_step.to_string();
    }

    std::ostream &operator<<(std::ostream &s, flux_y const) { return s << "initial step 2: "; }

    /*
     * The following operators and structs are for debugging only
     */
    std::ostream &operator<<(std::ostream &s, final_step const) { return s << "final step"; }

    bool test(uint_t x, uint_t y, uint_t t) {

        uint_t d1 = x;
        uint_t d2 = y;
        uint_t d3 = 1;

        //! [layout_map]

        //! [storage_type]
        typedef storage_traits<backend_t>::storage_info_t<0, 3> storage_info_t;
        typedef storage_traits<backend_t>::data_store_t<float_type, storage_info_t> sol_type;
        //! [storage_type]

        //! [proc_grid_dims]
        MPI_Comm CartComm;
        array<int, 3> dimensions{0, 0, 1};
        int period[3] = {1, 1, 1};
        MPI_Dims_create(PROCS, 2, &dimensions[0]);
        assert(dimensions[2] == 1);

        MPI_Cart_create(MPI_COMM_WORLD, 3, &dimensions[0], period, false, &CartComm);

        //! [proc_grid_dims]

        //! [pattern_type]
        typedef gridtools::halo_exchange_dynamic_ut<storage_info_t::layout_t,
            gridtools::layout_map<0, 1, 2>,
            float_type,
#ifdef __CUDACC__
            gridtools::gcl_gpu>
#else
            gridtools::gcl_cpu>
#endif
            pattern_type;

        pattern_type he(gridtools::boollist<3>(false, false, false), CartComm);
        //! [pattern_type]

        auto c_grid = he.comm();
        int pi, pj, pk;
        c_grid.coords(pi, pj, pk);
        assert(pk == 0);
        int di, dj, dk;
        c_grid.dims(di, dj, dk);
        assert(dk == 1);

        array<uint_t, 3> halo{1, 1, 0};

        //! [storage]
        storage_info_t storage_info(d1 + 2 * halo[0], d2 + 2 * halo[1], d3);
        //! [storage]

        sol_type h(storage_info);
        sol_type u(storage_info);
        sol_type v(storage_info);

        //! [add_halo]
        he.add_halo<0>(halo[0], halo[0], halo[0], d1 + halo[0] - 1, d1 + 2 * halo[0]);
        he.add_halo<0>(halo[1], halo[1], halo[1], d2 + halo[1] - 1, d2 + 2 * halo[1]);
        he.add_halo<0>(0, 0, 0, d3 - 1, d3);

        he.setup(3);
        //! [add_halo]

        //! [initialization_h]
        auto view_h = make_host_view(h);
        auto view_u = make_host_view(u);
        auto view_v = make_host_view(v);

        for (int i = 0; i < d1 + 2 * halo[0]; ++i) {
            for (int j = 0; j < d2 + 2 * halo[1]; ++j) {
                view_h(i, j, 0) = bc_periodic<0, 0>::droplet(i, j); // h
                view_u(i, j, 0) = 0.0;
                view_v(i, j, 0) = 0.0;
            }
        }

#ifndef NDEBUG
#ifndef __CUDACC__
        std::ofstream myfile;
        std::stringstream name;
        name << "example" << PID << ".txt";
        myfile.open(name.str().c_str());
#endif
#endif
        // Definition of the physical dimensions of the problem.
        // The constructor takes the horizontal plane dimensions,
        // while the vertical ones are set according the the axis property soon after
        //! [grid]
        auto grid = make_grid({halo[0], halo[0], halo[0], d1 + halo[0] - 1, d1 + 2 * halo[0]},
            {halo[1], halo[1], halo[1], d2 + halo[1] - 1, d2 + 2 * halo[1]},
            d3);
        //! [grid]

        // the following might be runtime value
        uint_t total_time = t;

        for (; final_step::current_time < total_time; ++final_step::current_time) {

            std::vector<float_type *> vec(3);

#ifdef __CUDACC__
            vec[0] = advanced::get_raw_pointer_of(make_device_view(h));
            vec[1] = advanced::get_raw_pointer_of(make_device_view(u));
            vec[2] = advanced::get_raw_pointer_of(make_device_view(v));
#else
            vec[0] = advanced::get_raw_pointer_of(make_host_view(h));
            vec[1] = advanced::get_raw_pointer_of(make_host_view(u));
            vec[2] = advanced::get_raw_pointer_of(make_host_view(v));
#endif

            // he.pack(vec);
            // he.exchange();
            // he.unpack(vec);
            //! [exchange]

#ifndef NDEBUG
#ifndef __CUDACC__
            h.sync();
            u.sync();
            v.sync();
            auto view00 = make_host_view(h);
            auto view10 = make_host_view(v);
            auto view20 = make_host_view(u);

            myfile << "INITIALIZED VALUES view00" << std::endl;
            for (int i = 0; i < d1 + 2 * halo[0]; ++i) {
                for (int j = 0; j < d2 + 2 * halo[1]; ++j) {
                    myfile << std::scientific << view00(i, j, 0) << " ";
                }
                myfile << "\n";
            }
            myfile << "\n";
            myfile << "INITIALIZED VALUES view10" << std::endl;
            for (int i = 0; i < d1 + 2 * halo[0]; ++i) {
                for (int j = 0; j < d2 + 2 * halo[1]; ++j) {
                    myfile << std::scientific << view10(i, j, 0) << " ";
                }
                myfile << "\n";
            }
            myfile << "\n";
            myfile << "INITIALIZED VALUES view20" << std::endl;
            for (int i = 0; i < d1 + 2 * halo[0]; ++i) {
                for (int j = 0; j < d2 + 2 * halo[1]; ++j) {
                    myfile << std::scientific << view20(i, j, 0) << " ";
                }
                myfile << "\n";
            }
            myfile << "\n";
            myfile << "#####################################################" << std::endl;
#endif
#endif

            // Definition of placeholders.
            //! [args]
            tmp_arg<0, float_type> p_hx;
            tmp_arg<1, float_type> p_ux;
            tmp_arg<2, float_type> p_vx;
            tmp_arg<3, float_type> p_hy;
            tmp_arg<4, float_type> p_uy;
            tmp_arg<5, float_type> p_vy;

            arg<0, sol_type> p_h;
            arg<1, sol_type> p_u;
            arg<2, sol_type> p_v;

            //! [run]
            compute<backend_t>(grid,
                p_h = h,
                p_u = u,
                p_v = v,
                make_multistage(execute::forward(),
                    make_stage<flux_x>(p_hx, p_ux, p_vx, p_h, p_u, p_v),
                    make_stage<flux_y>(p_hy, p_uy, p_vy, p_h, p_u, p_v),
                    make_stage<final_step>(p_hx, p_ux, p_vx, p_hy, p_uy, p_vy, p_h, p_u, p_v)));
            //! [run]
        }

        //! [finalize]
        he.wait();

        bool retval = true;
        //! [finalize]

#ifndef NDEBUG
#ifndef __CUDACC__
        myfile << "############## SOLUTION ################" << std::endl;
        auto view00 = make_host_view(h);
        auto view10 = make_host_view(v);
        auto view20 = make_host_view(u);
        myfile << "SOLUTION VALUES view00" << std::endl;
        for (int i = 0; i < d1 + 2 * halo[0]; ++i) {
            for (int j = 0; j < d2 + 2 * halo[1]; ++j) {
                myfile << std::scientific << view00(i, j, 0) << " ";
            }
            myfile << "\n";
        }
        myfile << "\n";
        myfile << "SOLUTION VALUES view10" << std::endl;
        for (int i = 0; i < d1 + 2 * halo[0]; ++i) {
            for (int j = 0; j < d2 + 2 * halo[1]; ++j) {
                myfile << std::scientific << view10(i, j, 0) << " ";
            }
            myfile << "\n";
        }
        myfile << "\n";
        myfile << "SOLUTION VALUES view20" << std::endl;
        for (int i = 0; i < d1 + 2 * halo[0]; ++i) {
            for (int j = 0; j < d2 + 2 * halo[1]; ++j) {
                myfile << std::scientific << view20(i, j, 0) << " ";
            }
            myfile << "\n";
        }
        myfile << "\n";
#endif

        verifier check_result(1e-8);
        array<array<uint_t, 2>, 3> halos{{{0, 0}, {0, 0}, {0, 0}}};
        shallow_water_reference<backend_t> reference(d1 + 2 * halo[0], d2 + 2 * halo[1]);

#ifndef __CUDACC__
        myfile << "############## REFERENCE INIT ################" << std::endl;
        view00 = make_host_view(reference.h);
        view10 = make_host_view(reference.u);
        view20 = make_host_view(reference.v);
        myfile << "REF INIT VALUES view00" << std::endl;
        for (int i = 0; i < d1 + 2 * halo[0]; ++i) {
            for (int j = 0; j < d2 + 2 * halo[1]; ++j) {
                myfile << std::scientific << view00(i, j, 0) << " ";
            }
            myfile << "\n";
        }
        myfile << "\n";
        myfile << "REF INIT VALUES view10" << std::endl;
        for (int i = 0; i < d1 + 2 * halo[0]; ++i) {
            for (int j = 0; j < d2 + 2 * halo[1]; ++j) {
                myfile << std::scientific << view10(i, j, 0) << " ";
            }
            myfile << "\n";
        }
        myfile << "\n";
        myfile << "REF INIT VALUES view20" << std::endl;
        for (int i = 0; i < d1 + 2 * halo[0]; ++i) {
            for (int j = 0; j < d2 + 2 * halo[1]; ++j) {
                myfile << std::scientific << view20(i, j, 0) << " ";
            }
            myfile << "\n";
        }
        myfile << "\n";
#endif

        for (uint_t t = 0; t < total_time; ++t) {
            reference.iterate();
        }

        retval &= check_result.verify(grid, reference.h, h, halos);

#ifndef __CUDACC__
        myfile << "############## REFERENCE ################" << std::endl;
        view00 = make_host_view(reference.h);
        view10 = make_host_view(reference.u);
        view20 = make_host_view(reference.v);

        myfile << "REF VALUES view00" << std::endl;
        for (int i = 0; i < d1 + 2 * halo[0]; ++i) {
            for (int j = 0; j < d2 + 2 * halo[1]; ++j) {
                myfile << std::scientific << view00(i, j, 0) << " ";
            }
            myfile << "\n";
        }
        myfile << "\n";
        myfile << "REF VALUES view10" << std::endl;
        for (int i = 0; i < d1 + 2 * halo[0]; ++i) {
            for (int j = 0; j < d2 + 2 * halo[1]; ++j) {
                myfile << std::scientific << view10(i, j, 0) << " ";
            }
            myfile << "\n";
        }
        myfile << "\n";
        myfile << "REF VALUES view20" << std::endl;
        for (int i = 0; i < d1 + 2 * halo[0]; ++i) {
            for (int j = 0; j < d2 + 2 * halo[1]; ++j) {
                myfile << std::scientific << view20(i, j, 0) << " ";
            }
            myfile << "\n";
        }
        myfile << "\n";
        myfile.close();
#endif
#endif
        std::cout << "shallow water parallel test SUCCESS?= " << std::boolalpha << retval << std::endl;
        return retval;
        //! [main]
    }

} // namespace shallow_water
