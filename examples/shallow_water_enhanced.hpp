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

// [includes]
#include <iostream>
#include <fstream>
#include <gridtools.hpp>
#include <stencil-composition/stencil-composition.hpp>
#include <storage/storage-facility.hpp>

#include <communication/halo_exchange.hpp>
#include "backend_select.hpp"

#include <tools/verifier.hpp>

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
using namespace enumtype;
using namespace expressions;
// [namespaces]

template < typename DX, typename DY, typename H >
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

        //! [dimension]
        typedef dimension< 5 > comp;
        //! [dimension]

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

    template < uint_t Component = 0, uint_t Snapshot = 0 >
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

        //! [accessor]
        typedef accessor< 1, enumtype::in, extent< 0, -1, 0, 0 >, 5 >
            sol; /** (input) is the solution at the cell center, computed at the previous time level */
        //! [accessor]
        typedef accessor< 0, enumtype::inout, extent< 0, 0, 0, 0 >, 5 >
            tmpx; /** (output) is the flux computed on the left edge of the cell */
        using arg_list = boost::mpl::vector2< tmpx, sol >;

        template < typename Evaluation >
        GT_FUNCTION static void Do(Evaluation &eval) {

            const float_type &tl = 2.;
            dimension< 1 > i;
            //![alias]
            using hx = alias< tmpx, comp >::set< 0 >;
            using h = alias< sol, comp >::set< 0 >;
            //![alias]
            using ux = alias< tmpx, comp >::set< 1 >;
            using u = alias< sol, comp >::set< 1 >;
            using vx = alias< tmpx, comp >::set< 2 >;
            using v = alias< sol, comp >::set< 2 >;

            //! [expression]
            eval(hx()) = eval((h() + h(i - 1)) / tl - (u() - u(i - 1)) * (dt() / (2 * dx())));
            //! [expression]

            eval(ux()) = eval((u() + u(i - 1)) / tl -
                              ((pow< 2 >(u()) / h() + pow< 2 >(h()) * g() / tl) -
                                  (pow< 2 >(u(i - 1)) / h(i - 1) + pow< 2 >(h(i - 1)) * (g() / tl))) *
                                  (dt() / (tl * dx())));

            eval(vx()) =
                eval((v() + v(i - 1)) / tl - (u() * v() / h() - u(i - 1) * v(i - 1) / h(i - 1)) * (dt() / (2 * dx())));
        }
    };
    // [flux_x]

    // [flux_y]
    struct flux_y : public functor_traits {

        typedef accessor< 0, enumtype::inout, extent< 0, 0, 0, 0 >, 5 >
            tmpy; /** (output) is the flux at the bottom edge of the cell */
        typedef accessor< 1, enumtype::in, extent< 0, 0, 0, -1 >, 5 >
            sol; /** (input) is the solution at the cell center, computed at the previous time level */
        using arg_list = boost::mpl::vector< tmpy, sol >;

        template < typename Evaluation >
        GT_FUNCTION static void Do(Evaluation &eval) {

            const float_type &tl = 2.;
            dimension< 2 > j;
            using h = alias< sol, comp >::set< 0 >;
            using hy = alias< tmpy, comp >::set< 0 >;
            using u = alias< sol, comp >::set< 1 >;
            using uy = alias< tmpy, comp >::set< 1 >;
            using v = alias< sol, comp >::set< 2 >;
            using vy = alias< tmpy, comp >::set< 2 >;

            eval(hy()) = eval((h() + h(j - 1)) / tl - (v() - v(j - 1)) * (dt() / (2 * dy())));

            eval(uy()) =
                eval((u() + u(j - 1)) / tl - (v() * u() / h() - v(j - 1) * u(j - 1) / h(j - 1)) * (dt() / (2 * dy())));

            eval(vy()) = eval((v() + v(j - 1)) / tl -
                              ((pow< 2 >(v()) / h() + pow< 2 >(h()) * g() / tl) -
                                  (pow< 2 >(v(j - 1)) / h(j - 1) + pow< 2 >(h(j - 1)) * (g() / tl))) *
                                  (dt() / (tl * dy())));
        }
    };
    // [flux_y]

    // [final_step]
    struct final_step : public functor_traits {

        typedef accessor< 0, enumtype::in, extent< 0, 1, 0, 1 >, 5 >
            tmpx; /** (input) is the flux at the left edge of the cell */
        typedef accessor< 1, enumtype::in, extent< 0, 1, 0, 1 >, 5 >
            tmpy; /** (input) is the flux at the bottom edge of the cell */
        typedef accessor< 2, enumtype::inout, extent< 0, 0, 0, 0 >, 5 >
            sol; /** (output) is the solution at the cell center, computed at the previous time level */
        typedef boost::mpl::vector< tmpx, tmpy, sol > arg_list;
        static uint_t current_time;

        //########## FINAL STEP #############
        // data dependencies with the previous parts
        // notation: alias<tmp, comp, step>::set<0, 0>() is ==> tmp(comp(0), step(0)).
        // Using a strategy to define some arguments beforehand

        template < typename Evaluation >
        GT_FUNCTION static void Do(Evaluation &eval) {
            const float_type &tl = 2.;
            dimension< 1 > i;
            dimension< 2 > j;
            using hx = alias< tmpx, comp >::set< 0 >;
            using h = alias< sol, comp >::set< 0 >;
            using hy = alias< tmpy, comp >::set< 0 >;
            using ux = alias< tmpx, comp >::set< 1 >;
            using u = alias< sol, comp >::set< 1 >;
            using uy = alias< tmpy, comp >::set< 1 >;
            using vx = alias< tmpx, comp >::set< 2 >;
            using v = alias< sol, comp >::set< 2 >;
            using vy = alias< tmpy, comp >::set< 2 >;

            eval(sol()) = eval(sol() - (ux(i + 1) - ux()) * (dt() / dx()) - (vy(j + 1) - vy()) * (dt() / dy()));

            eval(sol(comp(1))) = eval(sol(comp(1)) -
                                      (pow< 2 >(ux(i + 1)) / hx(i + 1) + hx(i + 1) * hx(i + 1) * ((g() / tl)) -
                                          (pow< 2 >(ux()) / hx() + pow< 2 >(hx()) * ((g() / tl)))) *
                                          ((dt() / dx())) -
                                      (vy(j + 1) * uy(j + 1) / hy(j + 1) - vy() * uy() / hy()) * (dt() / dy()));

            eval(sol(comp(2))) =
                eval(sol(comp(2)) - (ux(i + 1) * vx(i + 1) / hx(i + 1) - (ux() * vx()) / hx()) * ((dt() / dx())) -
                     (pow< 2 >(vy(j + 1)) / hy(j + 1) + pow< 2 >(hy(j + 1)) * ((g() / tl)) -
                         (pow< 2 >(vy()) / hy() + pow< 2 >(hy()) * ((g() / tl)))) *
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
        typedef backend_t::storage_traits_t::storage_info_t< 0, 3 > storage_info_t;
        typedef backend_t::storage_traits_t::data_store_field_t< float_type, storage_info_t, 1, 1, 1 > sol_type;
        //! [storage_type]

        // Definition of placeholders. The order of them reflects the order in which the user will deal with them
        // especially the non-temporary ones, in the construction of the domain
        //! [args]
        typedef tmp_arg< 0, sol_type > p_tmpx;
        typedef tmp_arg< 1, sol_type > p_tmpy;

        typedef arg< 2, sol_type > p_sol;

        //! [proc_grid_dims]
        array< int, 3 > dimensions{0, 0, 1};
        MPI_Dims_create(PROCS, 3, &dimensions[0]);

        //! [proc_grid_dims]

        //! [pattern_type]
        typedef gridtools::halo_exchange_dynamic_ut< typename storage_info_t::layout_t,
            gridtools::layout_map< 0, 1, 2 >,
            float_type,
            MPI_3D_process_grid_t< 3 >,
#ifdef __CUDACC__
            gridtools::gcl_gpu,
#else
            gridtools::gcl_cpu,
#endif
            gridtools::version_manual > pattern_type;

        pattern_type he(gridtools::boollist< 3 >(false, false, false), GCL_WORLD, dimensions);
        //! [pattern_type]

        auto c_grid = he.comm();
        int pi, pj, pk;
        c_grid.coords(pi, pj, pk);
        assert(pk == 0);
        int di, dj, dk;
        c_grid.dims(di, dj, dk);
        assert(dk == 1);

        array< ushort_t, 3 > halo{1, 1, 0};

        //! [storage]
        storage_info_t storage_info(d1 + 2 * halo[0], d2 + 2 * halo[1], d3);
        //! [storage]

        sol_type sol(storage_info);

        //! [add_halo]
        he.add_halo< 0 >(halo[0], halo[0], halo[0], d1 + halo[0] - 1, d1 + 2 * halo[0]);
        he.add_halo< 0 >(halo[1], halo[1], halo[1], d2 + halo[1] - 1, d2 + 2 * halo[1]);
        he.add_halo< 0 >(0, 0, 0, d3 - 1, d3);

        he.setup(3);
        //! [add_halo]

        //! [initialization_h]
        auto ds0 = sol.get< 0, 0 >();
        auto ds1 = sol.get< 1, 0 >();
        auto ds2 = sol.get< 2, 0 >();
        auto view0 = make_host_view(ds0);
        auto view1 = make_host_view(ds1);
        auto view2 = make_host_view(ds2);

        for (int i = 0; i < d1 + 2 * halo[0]; ++i) {
            for (int j = 0; j < d2 + 2 * halo[1]; ++j) {
                view0(i, j, 0) = bc_periodic< 0, 0 >::droplet(i, j); // h
                view1(i, j, 0) = 0.0;
                view2(i, j, 0) = 0.0;
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

        //! [computation]
        auto shallow_water_stencil = make_computation< backend_t >(
            grid,
            p_sol() = sol,
            make_multistage // mss_descriptor
            (execute< forward >(),
                make_independent(make_stage< flux_x >(p_tmpx(), p_sol()), make_stage< flux_y >(p_tmpy(), p_sol())),
                make_stage< final_step >(p_tmpx(), p_tmpy(), p_sol())));
        //! [computation]

        // the following might be runtime value
        uint_t total_time = t;

        for (; final_step::current_time < total_time; ++final_step::current_time) {

            std::vector< float_type * > vec(3);

#ifdef __CUDACC__
            vec[0] = advanced::get_initial_address_of(make_device_view(sol.get< 0, 0 >()));
            vec[1] = advanced::get_initial_address_of(make_device_view(sol.get< 1, 0 >()));
            vec[2] = advanced::get_initial_address_of(make_device_view(sol.get< 2, 0 >()));
#else
            vec[0] = advanced::get_initial_address_of(make_host_view(sol.get< 0, 0 >()));
            vec[1] = advanced::get_initial_address_of(make_host_view(sol.get< 1, 0 >()));
            vec[2] = advanced::get_initial_address_of(make_host_view(sol.get< 2, 0 >()));
#endif

// he.pack(vec);
// he.exchange();
// he.unpack(vec);
//! [exchange]

#ifndef NDEBUG
#ifndef __CUDACC__
            sol.sync();
            auto view = make_field_host_view(sol);
            auto view00 = view.get< 0, 0 >();
            auto view10 = view.get< 1, 0 >();
            auto view20 = view.get< 2, 0 >();

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

            //! [run]
            shallow_water_stencil.run();
            //! [run]
        }

        //! [finalize]
        he.wait();

        shallow_water_stencil.sync_all();

        bool retval = true;
//! [finalize]

#ifndef NDEBUG
#ifndef __CUDACC__
        myfile << "############## SOLUTION ################" << std::endl;
        auto view = make_field_host_view(sol);
        auto view00 = view.get< 0, 0 >();
        auto view10 = view.get< 1, 0 >();
        auto view20 = view.get< 2, 0 >();
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
        array< array< uint_t, 2 >, 3 > halos{{{0, 0}, {0, 0}, {0, 0}}};
        shallow_water_reference< backend_t > reference(d1 + 2 * halo[0], d2 + 2 * halo[1]);

#ifndef __CUDACC__
        myfile << "############## REFERENCE INIT ################" << std::endl;
        view = make_field_host_view(reference.solution);
        view00 = view.get< 0, 0 >();
        view10 = view.get< 1, 0 >();
        view20 = view.get< 2, 0 >();
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

        for (int i = 0; i < 1; ++i)
            retval &= check_result.verify(grid, reference.solution.get_field()[i], sol.get_field()[i], halos);

#ifndef __CUDACC__
        myfile << "############## REFERENCE ################" << std::endl;
        view = make_field_host_view(reference.solution);
        view00 = view.get< 0, 0 >();
        view10 = view.get< 1, 0 >();
        view20 = view.get< 2, 0 >();

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
