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
#include <common/halo_descriptor.hpp>
#include <boost/lambda/bind.hpp>
#include <boost/lambda/construct.hpp>
#include <boost/fusion/include/make_vector.hpp>

#include <stencil-composition/stencil-composition.hpp>

/*
  @file
  @brief This file shows an implementation of the "shallow water" stencil
  It defines
 */

using gridtools::level;
using gridtools::accessor;
using gridtools::extent;
using gridtools::arg;

using gridtools::direction;
using gridtools::sign;
using gridtools::minus_;
using gridtools::zero_;
using gridtools::plus_;

using namespace gridtools;
using namespace enumtype;
using namespace expressions;

namespace shallow_water {
    // This is the definition of the special regions in the "vertical" direction
    typedef gridtools::interval< level< 0, -1 >, level< 1, -1 > > x_interval;
    typedef gridtools::interval< level< 0, -2 >, level< 1, 1 > > axis;

    struct functor_traits {
        //#if  !((defined(__GNUC__)) && (__GNUC__ < 4) || (__GNUC__ == 4 && __GNUC_MINOR__ < 9))
        using tmp = arg_extend< accessor< 0, extent< -1, 1, -1, 1 > >, 2 >::type;
        using sol = arg_extend< accessor< 1, extent< -1, 1, -1, 1 > >, 2 >::type;
        using arg_list = boost::mpl::vector< tmp, sol >;
        using step = dimension< 3 >;
        using comp = dimension< 4 >;
        //#endif

        static float_type dx() { return 1e-2; }
        static float_type dy() { return 1e-2; }
        static float_type dt() { return 1e-3; }
        static float_type g() { return 9.81; }
    };

    struct bc_reflecting : functor_traits {
        // reflective boundary conditions in I and J
        template < sign I, sign J, typename DataField0, typename DataField1 >
        GT_FUNCTION void operator()(direction< I, J, zero_ >,
            DataField0 &data_field0,
            DataField1 const &data_field1,
            uint_t i,
            uint_t j,
            uint_t k) const {
            // TODO use placeholders here instead of the storage
            data_field0(i, j, k) = data_field1(i, j, k);
        }
    };

    // struct bc_vertical{
    //     // reflective boundary conditions in K
    //     template < sign K, typename DataField0, typename DataField1>
    //     GT_FUNCTION
    //     void operator()(direction<zero_, zero_, K>,
    //                     DataField0 & data_field0, DataField1 const & data_field1,
    //                     uint_t i, uint_t j, uint_t k) const {
    //         data_field0(i,j,k) = data_field1(i,j,k);
    //     }
    // };

    // These are the stencil operators that compose the multistage stencil in this test
    struct initial_step : public functor_traits {
/**GCC 4.8.2  bug: inheriting the 'using' aliases (or replacing the typedefs below with the 'using' syntax) from the
   base class produces an internal compiler error (segfault).
   The compilation runs fine without warnings with GCC >= 4.9 and Clang*/
#if (defined(__GNUC__)) && (__GNUC__ < 4) || (__GNUC__ == 4 && __GNUC_MINOR__ < 9)
        // shielding the base class aliases
        typedef arg_extend< accessor< 0, extent< -1, 1, -1, 1 > >, 2 >::type tmp;
        typedef arg_extend< accessor< 1, extent< -1, 1, -1, 1 > >, 2 >::type sol;
        typedef boost::mpl::vector< tmp, sol > arg_list;
        typedef dimension< 3 > step;
        typedef dimension< 4 > comp;
#endif
        /* static const auto expression=in(1,0,0)-out(); */

        template < typename Evaluation, typename ComponentU, typename DimensionX, typename DimensionY >
        GT_FUNCTION static float_type /*&&*/ half_step(
            Evaluation &eval, ComponentU U, DimensionX d1, DimensionY d2, float_type const &delta) {
            return /*std::move*/ (
                eval(sol(d1, d2) + sol(d2) / 2. - (sol(U, d2, d1) - sol(U, d2)) * (dt() / (2 * delta))));
        }

        template < typename Evaluation, typename ComponentU, typename DimensionX, typename DimensionY >
        GT_FUNCTION static float_type /*&&*/ half_step_u(
            Evaluation &eval, ComponentU U, DimensionX d1, DimensionY d2, float_type const &delta) {
            return /*std::move*/ (eval((
                sol(U, d1, d2) + sol(U, d2) / 2. -
                (pow< 2 >(sol(U, d1, d2)) / sol(d1, d2) + pow< 2 >(sol(d1, d2)) * g() / 2.) * (dt() / (2. * delta)) -
                pow< 2 >(sol(U, d2)) / sol(d2) - pow< 2 >(sol(d2)) * pow< 2 >(g() / 2.))));
        }

        template < typename Evaluation,
            typename ComponentU,
            typename ComponentV,
            typename DimensionX,
            typename DimensionY >
        GT_FUNCTION static float_type /*&&*/ half_step_v(
            Evaluation &eval, ComponentU U, ComponentV V, DimensionX d1, DimensionY d2, float_type const &delta) {
            return /*std::move*/ (eval((sol(V, d1, d2) + sol(V, d1) / 2. -
                                        sol(U, d1, d2) * sol(V, d1, d2) / sol(d1, d2) * (dt() / (2 * delta)) -
                                        sol(U, d2) * sol(V, d2) / sol(d2))));
        }

        template < typename Evaluation >
        GT_FUNCTION static void Do(Evaluation &eval, x_interval) {
            // x i;
            // y j;

            eval(/*tmp*/ out()) = 1.; // half_step  (eval, comp(1), x(1), y(1), dx());
            // eval(tmp(comp(1)))=1.;//half_step_u(eval, comp(1), x(1), y(1), dx());
            // eval(tmp(comp(2)))=1.;//half_step_v(eval, comp(1), comp(2), x(1), y(1), dx());

            // eval(tmp(comp(0), step(1)))=1.;//half_step  (eval, comp(2), y(1), x(1), dy());
            // eval(tmp(comp(1), step(1)))=1.;//half_step_v(eval, comp(2), comp(1), y(1), x(1), dy());
            // eval(tmp(comp(2), step(1)))=1.;//half_step_u(eval, comp(2), y(1), x(1), dy());
        }
    };

    struct final_step : public functor_traits {
#if (defined(__GNUC__)) && (__GNUC__ < 4) || (__GNUC__ == 4 && __GNUC_MINOR__ < 9)
        typedef arg_extend< accessor< 0, extent< -1, 1, -1, 1 > >, 2 >::type tmp;
        typedef arg_extend< accessor< 1, extent< -1, 1, -1, 1 > >, 2 >::type sol;
        typedef boost::mpl::vector< tmp, sol > arg_list;
        typedef dimension< 3 > step;
        typedef dimension< 4 > comp;
#endif
        template < typename Evaluation >
        GT_FUNCTION static void Do(Evaluation &eval, x_interval) {
            //########## FINAL STEP #############
            // data dependencies with the previous parts
            // notation: alias<tmp, comp, step>(0, 0) is ==> tmp(comp(0), step(0)).
            // Using a strategy to define some arguments beforehand

            x i;
            y j;
#ifdef __CUDACC__
            comp c;
            step s;

// eval(sol()) = eval(sol()-
//                    (tmp(c+1, i-1) - tmp(c+1, i-1, j-1))*(dt()/dx())-
//          tmp(c+2, s+1, j-1) - tmp(c+2, s+1, i-1, j-1)*(dt()/dy()));

// eval(sol(comp(1))) =  eval(sol(comp(1)) -
//                            (pow<2>(tmp(c+1, j-1))                / tmp(j-1)      + tmp(j-1)*tmp(j-1)*((g()/2.)) -
//     (pow<2>(tmp(c+1,i-1,j-1))            / tmp(i-1, j-1) +pow<2>(tmp(i-1,j-1) )*((g()/2.))))*((dt()/dx())) -
//           (tmp(c+2,s+1,i-1)*tmp(c+1,s+1,i-1)          / tmp(s+1,i-1) -
//     tmp(c+2,s+1,i-1, j-1)*tmp(c+1,s+1,i-1,j-1) / tmp(s+1,i-1, j-1) + tmp(s+1,i-1, j-1)*((g()/2.))) *((dt()/dy())));

// eval(sol(comp(2))) = eval(sol(comp(2)) -
//          (tmp(c+1,j-1)    *tmp(c+1,j-1)       /tmp(s+1,j-1) -
//                            (tmp(c+1,i-1,j-1)*tmp(c+2,i-1, j-1)) /tmp(i-1, j-1))*((dt()/dx()))-
//                           (pow<2>(tmp(c+2,s+1,i-1))                /tmp(s+1,i-1)      +pow<2>(tmp(s+1,i-1)
//                           )*((g()/2.)) -
//                            pow<2>(tmp(c+2,s+1,i-1,j-1))           /tmp(s+1,i-1,j-1) +pow<2>(tmp(s+1,i-1,
//                            j-1))*((g()/2.))   )*((dt()/dy())));
#else

            // eval(sol()) = eval(sol()-
            //                    (ux(x(-1)) - ux(x(-1), y(-1)))*(dt()/dx())-
            //                     vy(y(-1)) - vy(x(-1), y(-1))*(dt()/dy()));

            // eval(sol(comp(1))) =  eval(sol(comp(1)) -
            //           ((ux(y(-1))^2)               / hx(y(-1))      + hx(y(-1))*hx(y(-1))*((g()/2.)) -
            //           ((ux(x(-1),y(-1))^2)           / hx(x(-1), y(-1)) +(hx(x(-1),y(-1))
            //           ^2)*((g()/2.))))*((dt()/dx())) -
            //            (vy(x(-1))*uy(x(-1))          / hy(x(-1))                                                   -
            //      vy(x(-1), y(-1))*uy(x(-1),y(-1)) / hy(x(-1), y(-1)) + hy(x(-1), y(-1))*((g()/2.))) *((dt()/dy())));

            // eval(sol(comp(2))) = eval(sol(comp(2)) -
            //          (ux(y(-1))    *vx(y(-1))       /hy(y(-1)) -
            //                           (ux(x(-1),y(-1))*vx(x(-1), y(-1))) /hx(x(-1), y(-1)))*((dt()/dx()))-
            //                          ((vy(x(-1))^2)                /hy(x(-1))      +(hy(x(-1))     ^2)*((g()/2.)) -
            //                           (vy(x(-1), y(-1))^2)           /hy(x(-1), y(-1)) +(hy(x(-1),
            //                           y(-1))^2)*((g()/2.))   )*((dt()/dy())));

            auto hx = alias< tmp, comp, step >(0, 0);
            auto hy = alias< tmp, comp, step >(0, 1);
            auto ux = alias< tmp, comp, step >(1, 0);
            auto uy = alias< tmp, comp, step >(1, 1);
            auto vx = alias< tmp, comp, step >(2, 0);
            auto vy = alias< tmp, comp, step >(2, 1);

            eval(sol()) = eval(
                sol() - (ux(i - 1) - ux(i - 1, j - 1)) * (dt() / dx()) - vy(j - 1) - vy(i - 1, j - 1) * (dt() / dy()));

            eval(sol(comp(1))) =
                eval(sol(comp(1)) -
                     (pow< 2 >(ux(j - 1)) / hx(j - 1) + hx(j - 1) * hx(j - 1) * ((g() / 2.)) -
                         (pow< 2 >(ux(i - 1, j - 1)) / hx(i - 1, j - 1) + pow< 2 >(hx(i - 1, j - 1)) * ((g() / 2.)))) *
                         ((dt() / dx())) -
                     (vy(i - 1) * uy(i - 1) / hy(i - 1) - vy(i - 1, j - 1) * uy(i - 1, j - 1) / hy(i - 1, j - 1) +
                         hy(i - 1, j - 1) * ((g() / 2.))) *
                         ((dt() / dy())));

            eval(sol(comp(2))) =
                eval(sol(comp(2)) -
                     (ux(j - 1) * vx(j - 1) / hy(j - 1) - (ux(i - 1, j - 1) * vx(i - 1, j - 1)) / hx(i - 1, j - 1)) *
                         ((dt() / dx())) -
                     (pow< 2 >(vy(i - 1)) / hy(i - 1) + pow< 2 >(hy(i - 1)) * ((g() / 2.)) -
                         pow< 2 >(vy(i - 1, j - 1)) / hy(i - 1, j - 1) + pow< 2 >(hy(i - 1, j - 1)) * ((g() / 2.))) *
                         ((dt() / dy())));
#endif
        }
    };

    /*
     * The following operators and structs are for debugging only
     */
    std::ostream &operator<<(std::ostream &s, initial_step const) { return s << "initial step"; }

    /*
     * The following operators and structs are for debugging only
     */
    std::ostream &operator<<(std::ostream &s, final_step const) { return s << "final step"; }

    void handle_error(int_t) { std::cout << "error" << std::endl; }

    bool test(uint_t x, uint_t y, uint_t z) {
        {

            uint_t d1 = x;
            uint_t d2 = y;
            uint_t d3 = z;

#ifdef CUDA_EXAMPLE
#define BACKEND backend< Cuda, Block >
#else
#ifdef BACKEND_BLOCK
#define BACKEND backend< Host, Block >
#else
#define BACKEND backend< Host, Naive >
#endif
#endif
            //                      dims  z y x
            //                   strides xy x 1
            typedef gridtools::layout_map< 2, 1, 0 > layout_t;
            typedef gridtools::BACKEND::storage_type< float_type, layout_t >::type storage_type;

/* The nice interface does not compile today (CUDA 6.5) with nvcc (C++11 support not complete yet)*/
#ifdef __CUDACC__
            typedef base_storage< hybrid_pointer< float_type >, layout_t, false, 3 > base_type1;
            typedef storage_list< base_type1, 1 > extended_type;
            typedef data_field< extended_type, extended_type, extended_type > tmp_type;

            typedef base_storage< hybrid_pointer< float_type >, layout_t, false, 6 > base_type2;
            typedef storage_list< base_type2, 0 > extended_type2;
            typedef data_field< extended_type2, extended_type2, extended_type2 > sol_type;
#else
            typedef extend< storage_type::basic_type, 1, 1, 1 >::type tmp_type;
            typedef extend< storage_type::basic_type, 0, 0, 0 >::type sol_type;
#endif
            typedef tmp_type::original_storage::pointer_type ptr;

            // Definition of placeholders. The order of them reflect the order the user will deal with them
            // especially the non-temporary ones, in the construction of the domain
            typedef arg< 0, tmp_type > p_tmp;
            typedef arg< 1, sol_type > p_sol;
            typedef boost::mpl::vector< p_tmp, p_sol > accessor_list;

            // // Definition of the actual data fields that are used for input/output
            tmp_type tmp(d1, d2, d3);
            ptr out1(tmp.size()), out2(tmp.size()), out3(tmp.size()), out4(tmp.size()), out5(tmp.size()),
                out6(tmp.size());

            sol_type sol(d1, d2, d3);
            ptr out7(sol.size()), out8(sol.size()), out9(sol.size());

            tmp.set< 0, 0 >(out1);
            tmp.set< 1, 0 >(out2);
            tmp.set< 2, 0 >(out3);
            tmp.set< 0, 1 >(out4);
            tmp.set< 1, 1 >(out5);
            tmp.set< 2, 1 >(out6);

            sol.push_front< 0 >(out7, 1.); // h
            sol.push_front< 1 >(out8, 1.); // u
            sol.push_front< 2 >(out9, 1.); // v

            // sol.push_front<3>(out9, [](uint_t i, uint_t j, uint_t k) ->float_type {return 2.0*exp
            // (-5*(i^2+j^2));});//h

            // construction of the domain. The domain is the physical domain of the problem, with all the physical
            // fields that are used, temporary and not
            // It must be noted that the only fields to be passed to the constructor are the non-temporary.
            // The order in which they have to be passed is the order in which they appear scanning the placeholders in
            // order. (I don't particularly like this)
            gridtools::aggregator_type< accessor_list > domain(boost::fusion::make_vector(&tmp, &sol));

            // Definition of the physical dimensions of the problem.
            // The constructor takes the horizontal plane dimensions,
            // while the vertical ones are set according the the axis property soon after
            // gridtools::grid<axis> grid(2,d1-2,2,d2-2);
            halo_descriptor di{2, 2, 2, d1 - 3, d1};
            halo_descriptor dj{2, 2, 2, d2 - 3, d2};

            auto grid = make_grid(di, dj, axis< 1 >(d3));

            auto shallow_water_stencil = gridtools::make_computation< gridtools::BACKEND, layout_t >(
                domain,
                grid,
                gridtools::make_multistage // mss_descriptor
                (execute< forward >(),
                    gridtools::make_stage< initial_step >(p_tmp(), p_sol()),
                    gridtools::make_stage< final_step >(p_tmp(), p_sol())));

            shallow_water_stencil->ready();

            shallow_water_stencil->steady();

            gridtools::array< gridtools::halo_descriptor, 2 > halos;
            halos[0] = gridtools::halo_descriptor(1, 1, 1, d1 - 2, d1);
            halos[1] = gridtools::halo_descriptor(1, 1, 1, d2 - 2, d2);
            // TODO: use placeholders here instead of the storage
            // gridtools::boundary_apply< bc_reflecting<uint_t> >(halos, bc_reflecting<uint_t>()).apply(p_sol(),
            // p_sol());

            shallow_water_stencil->run();

            shallow_water_stencil->finalize();

            // sol.print();
        }
        return true;
    }

} // namespace shallow_water
