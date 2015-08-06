#pragma once

#include <gridtools.hpp>
#include <common/halo_descriptor.hpp>
#include <boost/lambda/bind.hpp>
#include <boost/lambda/construct.hpp>
#include <stencil-composition/interval.hpp>
#include <stencil-composition/make_computation.hpp>
#include <storage/parallel_storage.hpp>
#include <storage/partitioner_trivial.hpp>

#include <communication/halo_exchange.hpp>

#include <stencil-composition/backend.hpp>

#ifdef CUDA_EXAMPLE
#include <boundary-conditions/apply_gpu.hpp>
#else
#include <boundary-conditions/apply.hpp>
#endif

#include <communication/halo_exchange.hpp>

#define BACKEND_BLOCK 1
/*
  @file
  @brief This file shows an implementation of the "shallow water" stencil using the CXX03 interfaces, with periodic boundary conditions.
  It is an horrible crappy unreadable solution, but it is more portable for the moment
*/

using gridtools::level;
using gridtools::accessor;
using gridtools::range;
using gridtools::arg;

using gridtools::direction;
using gridtools::sign;
using gridtools::minus_;
using gridtools::zero_;
using gridtools::plus_;

using namespace gridtools;
using namespace enumtype;
using namespace expressions;

namespace shallow_water{
// This is the definition of the special regions in the "vertical" direction
    typedef interval<level<0,-1>, level<1,-1> > x_interval;
    typedef interval<level<0,-2>, level<1,1> > axis;

/**@brief This traits class defined the necessary typesand functions used by all the functors defining the shallow water model*/

    template<uint_t Component=0, uint_t Snapshot=0>
    struct bc_periodic {

        typedef dimension<5> comp;
        /**@brief space discretization step in direction i */
        GT_FUNCTION
        static float_type dx(){return 1.;}
        /**@brief space discretization step in direction j */
        GT_FUNCTION
        static float_type dy(){return 1.;}

        // periodic boundary conditions in I
        template <sign I, sign K, typename DataField0>
        GT_FUNCTION
        void operator()(direction<I, minus_, K, typename boost::enable_if_c<I!=minus_>::type>,
                        DataField0 & data_field0,
                        uint_t i, uint_t j, uint_t k) const {
            data_field0.template get<Component, Snapshot>()[data_field0._index(i,j,k)] = data_field0.template get<Component, Snapshot>()[data_field0._index(i,data_field0.template dims<1>()-1-j,k)];
        }

        // periodic boundary conditions in J
        template <sign J, sign K, typename DataField0>
        GT_FUNCTION
        void operator()(direction<minus_, J, K>,
                        DataField0 & data_field0,
                        uint_t i, uint_t j, uint_t k) const {
            data_field0.template get<Component, Snapshot>()[data_field0._index(i,j,k)] = data_field0.template get<Component, Snapshot>()[data_field0._index(data_field0.template dims<0>()-1-i,j,k)];
        }

        // default: do nothing
        template <sign I, sign J, sign K, typename P, typename DataField0>
        GT_FUNCTION
        void operator()(direction<I, J, K, P>,
                        DataField0 & data_field0,
                        uint_t i, uint_t j, uint_t k) const {
        }

        GT_FUNCTION
        static float_type droplet(uint_t const& i, uint_t const& j, uint_t const& k){
            return 1.+2. * std::exp(-5*((((int)i-4)*dx())*((((int)i-4)*dx()))+(((int)j-9)*dy())*(((int)j-9)*dy())));
        }

        GT_FUNCTION
        static float_type droplet_higher(uint_t const& i, uint_t const& j, uint_t const& k){
            return 1.+4. * std::exp(-5*((((int)i-3)*dx())*((((int)i-3)*dx()))+(((int)j-3)*dy())*(((int)j-3)*dy())));
        }

    };


    template<uint_t Component, uint_t Snapshot=0>
    struct bc_reflective;

    template<uint_t Snapshot>
    struct bc_reflective<0, Snapshot> {

        template <sign I, sign K, typename DataField0>
        GT_FUNCTION
        void operator()(direction<I, minus_, K>,
                        DataField0 & data_field0,
                        uint_t i, uint_t j, uint_t k) const {
            data_field0.template get<0, Snapshot>()[data_field0._index(i,j,k)] = data_field0.template get<0, Snapshot>()[data_field0._index(i,j+1,k)];
        }

        // periodic boundary conditions in I
        template <sign I, sign K, typename DataField0>
        GT_FUNCTION
        void operator()(direction<I, plus_, K, typename boost::enable_if_c<I!=minus_>::type>,
                        DataField0 & data_field0,
                        uint_t i, uint_t j, uint_t k) const {
            data_field0.template get<0, Snapshot>()[data_field0._index(i,j,k)] = data_field0.template get<0, Snapshot>()[data_field0._index(i,j-1,k)];
        }

        // periodic boundary conditions in J (I=0) for H
        template <sign J, sign K, typename DataField0>
        GT_FUNCTION
        void operator()(direction<minus_, J, K, typename boost::enable_if_c<J!=plus_&&J!=minus_>::type>,
                        DataField0 & data_field0,
                        uint_t i, uint_t j, uint_t k) const {
            data_field0.template get<0, Snapshot>()[data_field0._index(i,j,k)] = data_field0.template get<0, Snapshot>()[data_field0._index(i+1,j,k)];
        }

        // periodic boundary conditions in J (I=0) for H
        template <sign J, sign K, typename DataField0>
        GT_FUNCTION
        void operator()(direction<plus_, J, K, typename boost::enable_if_c<J!=minus_&&J!=plus_>::type>,
                        DataField0 & data_field0,
                        uint_t i, uint_t j, uint_t k) const {
            data_field0.template get<0, Snapshot>()[data_field0._index(i,j,k)] = data_field0.template get<0, Snapshot>()[data_field0._index(i-1,j,k)];
        }

        // default: do nothing
        template <sign I, sign J, sign K, typename P, typename DataField0>
        GT_FUNCTION
        void operator()(direction<I, J, K, P>,
                        DataField0 & data_field0,
                        uint_t i, uint_t j, uint_t k) const {
        }

    };

    template<uint_t Snapshot>
    struct bc_reflective<1, Snapshot> {

        // periodic boundary conditions in I (J=0)
        template <sign I, sign K, typename DataField0>
        GT_FUNCTION
        void operator()(direction<I, minus_, K>,
                        DataField0 & data_field0,
                        uint_t i, uint_t j, uint_t k) const {
            data_field0.template get<1, Snapshot>()[data_field0._index(i,j,k)] = data_field0.template get<1, Snapshot>()[data_field0._index(i,j+1,k)];
        }

        // periodic boundary conditions in I (J=N)
        template <sign I, sign K, typename DataField0>
        GT_FUNCTION
        void operator()(direction<I, plus_, K, typename boost::enable_if_c<I!=minus_>::type>,
                        DataField0 & data_field0,
                        uint_t i, uint_t j, uint_t k) const {
            data_field0.template get<1, Snapshot>()[data_field0._index(i,j,k)] = data_field0.template get<1, Snapshot>()[data_field0._index(i,j-1,k)];
        }

        // periodic boundary conditions in J (I=0) for H
        template <sign J, sign K, typename DataField0>
        GT_FUNCTION
        void operator()(direction<minus_, J, K, typename boost::enable_if_c<J!=minus_&&J!=plus_>::type>,
                        DataField0 & data_field0,
                        uint_t i, uint_t j, uint_t k) const {
            data_field0.template get<1, Snapshot>()[data_field0._index(i,j,k)] = -data_field0.template get<1, Snapshot>()[data_field0._index(i+1,j,k)];
        }

        // periodic boundary conditions in J (I=0) for H
        template <sign J, sign K, typename DataField0, typename boost::enable_if_c<J!=minus_&&J!=plus_>::type>
            GT_FUNCTION
            void operator()(direction<plus_, J, K, typename boost::enable_if_c<J!=minus_&&J!=plus_>::type>,
                            DataField0 & data_field0,
                            uint_t i, uint_t j, uint_t k) const {
            data_field0.template get<1, Snapshot>()[data_field0._index(i,j,k)] = -data_field0.template get<1, Snapshot>()[data_field0._index(i-1,j,k)];
        }

        // default: do nothing
        template <sign I, sign J, sign K, typename P, typename DataField0>
        GT_FUNCTION
        void operator()(direction<I, J, K, P>,
                        DataField0 & data_field0,
                        uint_t i, uint_t j, uint_t k) const {
        }
    };

    template<uint_t Snapshot>
    struct bc_reflective<2, Snapshot> {

        // periodic boundary conditions in I
        template <sign I, sign K, typename DataField0>
        GT_FUNCTION
        void operator()(direction<I, minus_, K>,
                        DataField0 & data_field0,
                        uint_t i, uint_t j, uint_t k) const {
            data_field0.template get<2, Snapshot>()[data_field0._index(i,j,k)] = -data_field0.template get<2, Snapshot>()[data_field0._index(i,j+1,k)];
        }

        // periodic boundary conditions in I
        template <sign I, sign K, typename DataField0>
        GT_FUNCTION
        void operator()(direction<I, plus_, K, typename boost::enable_if_c<I!=minus_>::type>,
                        DataField0 & data_field0,
                        uint_t i, uint_t j, uint_t k) const {
            data_field0.template get<2, Snapshot>()[data_field0._index(i,j,k)] = -data_field0.template get<2, Snapshot>()[data_field0._index(i,j-1,k)];
        }

        // periodic boundary conditions in J (I=0) for H
        template <sign J, sign K, typename DataField0>
        GT_FUNCTION
        void operator()(direction<minus_, J, K, typename boost::enable_if_c<J!=minus_&&J!=plus_>::type>,
                        DataField0 & data_field0,
                        uint_t i, uint_t j, uint_t k) const {
            data_field0.template get<2, Snapshot>()[data_field0._index(i,j,k)] = data_field0.template get<2, Snapshot>()[data_field0._index(i+1,j,k)];
        }

        // periodic boundary conditions in J (I=0) for H
        template <sign J, sign K, typename DataField0>
        GT_FUNCTION
        void operator()(direction<plus_, J, K, typename boost::enable_if_c<J!=minus_&&J!=plus_>::type>,
                        DataField0 & data_field0,
                        uint_t i, uint_t j, uint_t k) const {
            data_field0.template get<2, Snapshot>()[data_field0._index(i,j,k)] = data_field0.template get<2, Snapshot>()[data_field0._index(i-1,j,k)];
        }

        // default: do nothing
        template <sign I, sign J, sign K, typename P, typename DataField0>
        GT_FUNCTION
        void operator()(direction<I, J, K, P>,
                        DataField0 & data_field0,
                        uint_t i, uint_t j, uint_t k) const {
        }
    };

// These are the stencil operators that compose the multistage stencil in this test
    struct first_step_x {

        // using xrange=range<0,-2,0,-2>;
        // using xrange_subdomain=range<0,1,0,0>;
        typedef accessor<0, range<0, 0, 0, 0>, 5> tmpx; /** (output) is the flux computed on the left edge of the cell */
        typedef accessor<1, range<0, +1, 0, +1>, 5> sol; /** (input) is the solution at the cell center, computed at the previous time level */
        typedef boost::mpl::vector<tmpx, sol> arg_list;

        typedef dimension<5> comp;
        /* *@brief space discretization step in direction i */
        GT_FUNCTION
        static float_type dx(){return 1.;}
        /**@brief space discretization step in direction j */
        GT_FUNCTION
        static float_type dy(){return 1.;}
        /**@brief time discretization step */
        GT_FUNCTION
        static float_type dt(){return .02;}
        /**@brief gravity acceleration */
        GT_FUNCTION
        static float_type g(){return 9.81;}

        // static const x::Index i;
        // static const y::Index j;

        template <typename Evaluation>
        GT_FUNCTION
        static void Do(Evaluation const & eval, x_interval) {

            x::Index i;
            y::Index j;
            // eval(tmpx(comp(1), x(-1), y(-2)))=eval(sol(comp(+5), x(+6), y(+7)))+3;
            eval(tmpx())= (eval(sol(comp(0),i+1,j+1)) +eval(sol(comp(0),j+1)))/2. -
                (eval(sol(comp(1),i+1,j+1)) - eval(sol(comp(1),j+1)))*(dt()/(2*dx()));

            eval(tmpx(comp(1)))=(eval(sol(comp(1),i+1, j+1)) +
                                 eval(sol(comp(1),j+1)))/2.-
                (((eval(sol(comp(1),i+1,j+1))*eval(sol(comp(1),i+1,j+1)))/eval(sol(comp(0),i+1,j+1))+(eval(sol(comp(0),i+1,j+1))*eval(sol(comp(0),i+1,j+1)))*g()/2.)  -
                 ((eval(sol(comp(1),j+1))*eval(sol(comp(1),j+1)))/eval(sol(comp(0),j+1)) +
                  (eval(sol(comp(0),j+1))*eval(sol(comp(0),j+1)))*(g()/2.)
                     ))*(dt()/(2.*dx()));

            eval(tmpx(comp(2)))= (eval(sol(comp(2),i+1,j+1)) +
                                  eval(sol(comp(2),j+1)))/2. -
                (eval(sol(comp(1),i+1,j+1))*eval(sol(comp(2),i+1,j+1))/eval(sol(comp(0),i+1,j+1)) -
                 eval(sol(comp(1),j+1))*eval(sol(comp(2),j+1))/eval(sol(comp(0),j+1)))*(dt()/(2*dx())) ;
        }
    };

    struct second_step_y {

        typedef dimension<5> comp;
        /**@brief space discretization step in direction i */
        GT_FUNCTION
        static float_type dx(){return 1.;}
        /**@brief space discretization step in direction j */
        GT_FUNCTION
        static float_type dy(){return 1.;}
        /**@brief time discretization step */
        GT_FUNCTION
        static float_type dt(){return .02;}
        /**@brief gravity acceleration */
        GT_FUNCTION
        static float_type g(){return 9.81;}

        // using xrange=range<0,-2,0,-2>;
        // using xrange_subdomain=range<0,0,0,1>;

        typedef accessor<0,range<0, 0, 0, 0>, 5> tmpy; /** (output) is the flux at the bottom edge of the cell */
        typedef accessor<1,range<0, +1, 0, +1>, 5> sol; /** (input) is the solution at the cell center, computed at the previous time level */
        typedef boost::mpl::vector<tmpy, sol> arg_list;

        template <typename Evaluation>
        GT_FUNCTION
        static void Do(Evaluation const & eval, x_interval) {

            x::Index i;
            y::Index j;

            eval(tmpy(comp(0)))= (eval(sol(comp(0),i+1,j+1)) + eval(sol(comp(0),i+1)))/2. -
                (eval(sol(comp(2),i+1,j+1)) - eval(sol(comp(2),i+1)))*(dt()/(2*dy()));

            eval(tmpy(comp(1)))=(eval(sol(comp(1),i+1,j+1)) +
                                 eval(sol(comp(1),i+1)))/2. -
                (eval(sol(comp(2),i+1,j+1))*eval(sol(comp(1),i+1,j+1))/eval(sol(comp(0),i+1,j+1)) -
                 eval(sol(comp(2),i+1))*eval(sol(comp(1),i+1))/eval(sol(comp(0),i+1)))*(dt()/(2*dy()));

            eval(tmpy(comp(2)))=(eval(sol(comp(2),i+1, j+1)) +
                                 eval(sol(comp(2),i+1)))/2.-
                (((eval(sol(comp(2),i+1,j+1))*eval(sol(comp(2),i+1,j+1)))/eval(sol(comp(0),i+1,j+1))+(eval(sol(comp(0),i+1,j+1))*eval(sol(comp(0),i+1,j+1)))*g()/2.)  -
                 (eval(sol(comp(2),i+1))*eval(sol(comp(2),i+1))/eval(sol(comp(0),i+1)) +
                  (eval(sol(comp(0),i+1))*eval(sol(comp(0),i+1)))*(g()/2.)
                     ))*(dt()/(2.*dy()));

        }
    };
    // const x::Index second_step_y::i;
    // const y::Index second_step_y::j;

    struct final_step {

//         typedef range<0,-3,0,-2> xrange;
        /* typedef range<0,-3,0,-3> xrange; */
        /* typedef range<1,1,1,1> xrange_subdomain; */

        typedef dimension<5> comp;
        /**@brief space discretization step in direction i */
        GT_FUNCTION
        static float_type dx(){return 1.;}
        /**@brief space discretization step in direction j */
        GT_FUNCTION
        static float_type dy(){return 1.;}
        /**@brief time discretization step */
        GT_FUNCTION
        static float_type dt(){return .02;}
        /**@brief gravity acceleration */
        GT_FUNCTION
        static float_type g(){return 9.81;}

        // using xrange=range<0,-2,0,-2>;
        // using xrange_subdomain=range<1,1,1,1>;

        typedef accessor<0, range<0,0,0,0>, 5> tmpx; /** (input) is the flux at the left edge of the cell */
        typedef accessor<1, range<0,0,0,0>, 5> tmpy; /** (input) is the flux at the bottom edge of the cell */
        typedef accessor<2,range<-1, 0, -1, 0>, 5> sol; /** (output) is the solution at the cell center, computed at the previous time level */
        typedef boost::mpl::vector<tmpx, tmpy, sol> arg_list;
        static uint_t current_time;

        //########## FINAL STEP #############
        //data dependencies with the previous parts
        //notation: alias<tmp, comp, step>(0, 0) is ==> tmp(comp(0), step(0)).
        //Using a strategy to define some arguments beforehand

        template <typename Evaluation>
        GT_FUNCTION
        static void Do(Evaluation const & eval, x_interval) {

            x::Index i;
            y::Index j;

            eval(sol()) =
                //h
                eval(sol())-
                //(ux(j-1)                -          ux(i-1, j-1) )(dt/dx)
                (eval(tmpx(comp(1),j-1)) - eval(tmpx(comp(1),i-1, j-1)))*(dt()/dx())
                -
                //(vy(i-1)               -           vy(i-1, j-1) )(dt/dy)
                (eval(tmpy(comp(2),i-1)) - eval(tmpy(comp(2),i-1, j-1)))*(dt()/dy())
                ;

            eval(sol(comp(1))) =
                eval(sol(comp(1))) -
                //(     ux(j-1)*ux(j-1)                                           / hx(j-1) )                    +      hx(j-1)           *     hx(j-1)          *(g/2)                       -
                ((eval(tmpx(comp(1),j-1))*eval(tmpx(comp(1),j-1)))                / eval(tmpx(comp(0),j-1))      + eval(tmpx(comp(0),j-1))*eval(tmpx(comp(0),j-1))*((g()/2.))                 -
                 //     ux(i-1, j-1)              ux(i-1,j-1)                         /hx(i-1, j-1)                   +     h(i-1,j-1)             *    h(i-1, j-1)*(g/2)
                 ((eval(tmpx(comp(1),i-1,j-1))*eval(tmpx(comp(1),i-1,j-1)))            / eval(tmpx(comp(0),i-1, j-1)) +(eval(tmpx(comp(0),i-1,j-1))*eval(tmpx(comp(0),i-1,j-1)) )*((g()/2.))))*((dt()/dx())) -
            //(    vy(i-1)          *     uy(i-1)                     /      hy(i-1)
            (eval(tmpy(comp(2),i-1))*eval(tmpy(comp(1),i-1))          / eval(tmpy(comp(0),i-1))                                                   -
             //    vy(i-1, j-1)          *      uy(i-1, j-1)          /       hy(i-1, j-1))dt/dy
             eval(tmpy(comp(2),i-1, j-1))*eval(tmpy(comp(1),i-1,j-1)) / eval(tmpy(comp(0),i-1, j-1))) *(dt()/dy())
                ;

            eval(sol(comp(2))) =
                // v()
                eval(sol(comp(2))) -
                // (ux(j-1)*vx(j-1)                                         /      hx(j-1) )        -
                (eval(tmpx(comp(1),j-1))    *eval(tmpx(comp(2),j-1))       /eval(tmpx(comp(0),j-1)) -
                 //      ux(i-1,j-1)*vx(i-1,j-1)                            /      hx(i-1,j-1))           * dt/dx       -
                 (eval(tmpx(comp(1),i-1,j-1))*eval(tmpx(comp(2),i-1, j-1))) /eval(tmpx(comp(0),i-1, j-1)))*((dt()/dx()))-
                ((eval(tmpy(comp(2),i-1))*eval(tmpy(comp(2),i-1)))                /eval(tmpy(comp(0),i-1))      +(eval(tmpy(comp(0),i-1))*eval(tmpy(comp(0),i-1))     )*((g()/2.)) -
                 ((eval(tmpy(comp(2),i-1, j-1))*eval(tmpy(comp(2),i-1, j-1)))     /eval(tmpy(comp(0),i-1, j-1)) +(eval(tmpy(comp(0),i-1, j-1))*eval(tmpy(comp(0),i-1, j-1)))*((g()/2.))   ))*((dt()/dy()));

        }

    };


    uint_t final_step::current_time=0;

/*
 * The following operators and structs are for debugging only
 */
    std::ostream& operator<<(std::ostream& s, first_step_x const) {
        return s << "initial step 1: ";
    }

    std::ostream& operator<<(std::ostream& s, second_step_y const) {
        return s << "initial step 2: ";
    }

/*
 * The following operators and structs are for debugging only
 */
    std::ostream& operator<<(std::ostream& s, final_step const) {
        return s << "final step";
    }

    bool test(uint_t x, uint_t y, uint_t z, uint_t t) {

        uint_t d1 = x;
        uint_t d2 = y;
        uint_t d3 = z;

#ifdef CUDA_EXAMPLE
#define BACKEND backend<Cuda, Block >
#else
#ifdef BACKEND_BLOCK
#define BACKEND backend<Host, Block >
#else
#define BACKEND backend<Host, Naive >
#endif
#endif
        //                      dims  z y x
        //                   strides xy x 1
        typedef layout_map<2,1,0> layout_t;
        typedef gridtools::BACKEND::storage_type<float_type, layout_t >::type storage_type;
        typedef gridtools::BACKEND::temporary_storage_type<float_type, layout_t >::type tmp_storage_type;

        /* The nice interface does not compile today (CUDA 6.5) with nvcc (C++11 support not complete yet)*/
        typedef field<storage_type, 1, 1, 1>::type sol_type;
        typedef field</*tmp_*/storage_type, 1, 1, 1>::type tmp_type;
        typedef sol_type::original_storage::pointer_type ptr;

        // Definition of placeholders. The order of them reflects the order the user will deal with them
        // especially the non-temporary ones, in the construction of the domain
        typedef arg<0, tmp_type > p_tmpx;
        typedef arg<1, tmp_type > p_tmpy;
        typedef arg<2, sol_type > p_sol;
        typedef boost::mpl::vector<p_tmpx, p_tmpy, p_sol> accessor_list;
        typedef sol_type::original_storage::pointer_type pointer_type;

        typedef gridtools::halo_exchange_dynamic_ut<layout_t,
                                                    gridtools::layout_map<0, 1, 2>,
                                                    pointer_type::pointee_t, MPI_3D_process_grid_t<3>,
#ifdef __CUDACC__
                                                    gridtools::gcl_gpu,
#else
                                                    gridtools::gcl_cpu,
#endif
                                                    gridtools::version_manual> pattern_type;

        gridtools::array<int, 3> dimensions(0,0,0);
        MPI_3D_process_grid_t<3>::dims_create(PROCS, 2, dimensions);
        dimensions[2]=1;

        pattern_type he(gridtools::boollist<3>(false,false,false), GCL_WORLD, &dimensions);

        array<ushort_t, 3> padding(1,1,0);
        array<ushort_t, 3> halo(1,1,0);
        typedef partitioner_trivial< cell_topology<topology::cartesian<layout_map<0,1,2> > > , pattern_type::grid_type> partitioner_t;
        partitioner_t part(he.comm(), halo, padding);
        parallel_storage<sol_type, partitioner_t> sol(part);
        sol.setup(d1, d2, d3);
        sol.allocate();
        parallel_storage<sol_type, partitioner_t> tmpx(part);
        tmpx.setup(d1, d2, d3);
        tmpx.allocate();
        parallel_storage<sol_type, partitioner_t> tmpy(part);
        tmpy.setup(d1, d2, d3);
        tmpy.allocate();

        he.add_halo<0>(sol.get_halo_gcl<0>());
        he.add_halo<1>(sol.get_halo_gcl<1>());
        he.add_halo<2>(0, 0, 0, d3 - 1, d3);

        he.setup(3);

        if(!he.comm().pid())
            sol.set<0,0>( &bc_periodic<0,0>::droplet);//h
        else
            sol.set<0,0>( &bc_periodic<0,0>::droplet_higher);//h
        sol.set<0,1>( 0.);//u
        sol.set<0,2>( 0.);//v

#ifndef NDEBUG
        std::ofstream myfile;
        std::stringstream name;
        name<<"example"<<PID<<".txt";
        myfile.open (name.str().c_str());
#endif

//         std::cout<<"INITIALIZED VALUES"<<std::endl;
//         sol.print(myfile);
//         std::cout<<"#####################################################"<<std::endl;

        // construction of the domain. The domain is the physical domain of the problem, with all the physical fields that are used, temporary and not
        // It must be noted that the only fields to be passed to the constructor are the non-temporary.
        // The order in which they have to be passed is the order in which they appear scanning the placeholders in order. (I don't particularly like this)
        domain_type<accessor_list> domain
            (boost::fusion::make_vector(&tmpx, &tmpy, &sol));

        // Definition of the physical dimensions of the problem.
        // The constructor takes the horizontal plane dimensions,
        // while the vertical ones are set according the the axis property soon after
        // coordinates<axis> coords(2,d1-2,2,d2-2);
        coordinates<axis, partitioner_t> coords(part, sol);

        coords.value_list[0] = 0;
        coords.value_list[1] = d3-1;

#ifdef __CUDACC__
        gridtools::computation*
#else
            boost::shared_ptr<gridtools::computation>
#endif
            shallow_water_stencil =
            make_computation<gridtools::BACKEND, layout_t>
            (
                make_mss // mss_descriptor
                (
                    execute<forward>(),
                    make_independent(
                        make_esf<first_step_x> (p_tmpx(), p_sol() ),
                        make_esf<second_step_y>(p_tmpy(), p_sol() )),
                    make_esf<final_step>(p_tmpx(), p_tmpy(), p_sol() )
                    ),
                domain, coords
                );

        shallow_water_stencil->ready();

        shallow_water_stencil->steady();

        uint_t total_time=t;

        for (;final_step::current_time < total_time; ++final_step::current_time)
        {
#ifdef CUDA_EXAMPLE
            /*                        component,snapshot */
//             boundary_apply_gpu< bc_reflective<0,0> >(halos, bc_reflective<0,0>()).apply(sol);
//             boundary_apply_gpu< bc_reflective<1,0> >(halos, bc_reflective<1,0>()).apply(sol);
//             boundary_apply_gpu< bc_reflective<2,0> >(halos, bc_reflective<2,0>()).apply(sol);
#else
            /*                    component,snapshot */
//             boundary_apply< bc_reflective<0,0> >(halos, bc_reflective<0,0>()).apply(sol);
//             boundary_apply< bc_reflective<1,0> >(halos, bc_reflective<1,0>()).apply(sol);
//             boundary_apply< bc_reflective<2,0> >(halos, bc_reflective<2,0>()).apply(sol);
#endif


#ifndef CUDA_EXAMPLE
                boost::timer::cpu_timer time;
#endif
                shallow_water_stencil->run();
#ifndef CUDA_EXAMPLE
                boost::timer::cpu_times lapse_time = time.elapsed();
                if(PID==0)
                    std::cout << "TIME " << boost::timer::format(lapse_time) << std::endl;
#endif

            std::vector<pointer_type::pointee_t*> vec(3);
            vec[0]=sol.fields()[0].get();
            vec[1]=sol.fields()[1].get();
            vec[2]=sol.fields()[2].get();
//             vec[3]=tmpx.fields()[0].get();
//             vec[4]=tmpx.fields()[1].get();
//             vec[5]=tmpx.fields()[2].get();
//             vec[6]=tmpy.fields()[0].get();
//             vec[7]=tmpy.fields()[1].get();
//             vec[8]=tmpy.fields()[2].get();

            he.pack(vec);
            he.exchange();
            he.unpack(vec);

#ifndef NDEBUG
            shallow_water_stencil->finalize();
            sol.print(myfile);
            //tmpx.print(myfile);
            //tmpy.print(myfile);
#endif
        }

#ifdef NDEBUG
        shallow_water_stencil->finalize();
#else
        myfile.close();
#endif

        // hdf5_driver<decltype(sol)> out("out.h5", "h", sol);
        // out.write(sol.get<0,0>());

        he.wait();

        return true;

    }

}//namespace shallow_water
