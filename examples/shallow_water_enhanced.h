#pragma once

#include <gridtools.h>
#include <stencil-composition/make_computation.h>
#include <storage/parallel_storage.h>
#include <storage/partitioner_trivial.h>
#include <stencil-composition/backend.h>

#ifdef CUDA_EXAMPLE
#include <boundary-conditions/apply_gpu.h>
#else
#include <boundary-conditions/apply.h>
#endif

#include <communication/halo_exchange.h>

//#define BACKEND_BLOCK 1
/*
  @file
  @brief This file shows an implementation of the "shallow water" stencil, with periodic boundary conditions

  For an exhaustive description of the shallow water problem refer to: http://www.mathworks.ch/moler/exm/chapters/water.pdf

  NOTE: It is the most human readable and efficient solution among the versions implemented, but it must be compiled for the host, with Clang or GCC>=4.9, and with C++11 enabled
 */

using gridtools::level;
using gridtools::arg_type;
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
    struct functor_traits{
        using comp=Dimension<5>;

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

        static x::Index i;
        static y::Index j;
        typedef decltype(i) i_t;        typedef decltype(j) j_t;

    };

    functor_traits::i_t functor_traits::i;
    functor_traits::j_t functor_traits::j;

    template<uint_t Component=0, uint_t Snapshot=0>
    struct bc_periodic : functor_traits {
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

        static constexpr float_type height=2.;
        GT_FUNCTION
        static float_type droplet(uint_t const& i, uint_t const& j, uint_t const& k){
                return 1.+height * std::exp(-5*(((i-15)*dx())*(((i-15)*dx()))+((j-15)*dy())*((j-15)*dy())));
       }
};

// These are the stencil operators that compose the multistage stencil in this test
    struct first_step_x        : public functor_traits {

        using xrange=range<0,-2,0,-2>;
        using xrange_subdomain=range<0,1,0,0>;
        typedef arg_type<0, range<0, 0, 0, 0>, 5> tmpx;
        typedef arg_type<1, range<0, 0, 0, 0>, 5> sol;
        using arg_list=boost::mpl::vector<tmpx, sol> ;

        template <typename Evaluation>
        GT_FUNCTION
        static void Do(Evaluation const & eval, x_interval) {

            using hx=alias<tmpx, comp>::set<0>; using h=alias<sol, comp>::set<0>;
            using ux=alias<tmpx, comp>::set<1>; using u=alias<sol, comp>::set<1>;
            using vx=alias<tmpx, comp>::set<2>; using v=alias<sol, comp>::set<2>;

            eval(hx())=
                eval((h(i+1,j+1) +h(j+1))/2. -
                     (u(i+1,j+1) - u(j+1))*(dt()/(2*dx())));

            eval(ux())=
                eval((u(i+1, j+1) +
                      u(j+1))/2.-
                     ((pow<2>(u(i+1,j+1))/h(i+1,j+1)+pow<2>(h(i+1,j+1))*g()/2.)  -
                      (pow<2>(u(j+1))/h(j+1) +
                       pow<2>(h(j+1))*(g()/2.)
                          ))*(dt()/(2.*dx())));

            eval(vx())=
                eval( (v(i+1,j+1) +
                       v(j+1))/2. -
                      (u(i+1,j+1)*v(i+1,j+1)/h(i+1,j+1) -
                       u(j+1)*v(j+1)/h(j+1))*(dt()/(2*dx())) );

        }
    };


    struct second_step_y        : public functor_traits {

        using xrange=range<0,-2,0,-2>;
        using xrange_subdomain=range<0,0,0,1>;

        typedef arg_type<0,range<0, 0, 0, 0>, 5> tmpy;
        typedef arg_type<1,range<0, 0, 0, 0>, 5> sol;
        using arg_list=boost::mpl::vector<tmpy, sol> ;

        template <typename Evaluation>
        GT_FUNCTION
        static void Do(Evaluation const & eval, x_interval) {

            using h=alias<sol, comp>::set<0>; using hy=alias<tmpy, comp>::set<0>;
            using u=alias<sol, comp>::set<1>; using uy=alias<tmpy, comp>::set<1>;
            using v=alias<sol, comp>::set<2>; using vy=alias<tmpy, comp>::set<2>;


            eval(hy())= eval((h(i+1,j+1) + h(i+1))/2. -
                             (v(i+1,j+1) - v(i+1))*(dt()/(2*dy())) );

            eval(uy())=eval( (u(i+1,j+1) +
                              u(i+1))/2. -
                             (v(i+1,j+1)*u(i+1,j+1)/h(i+1,j+1) -
                              v(i+1)*u(i+1)/h(i+1))*(dt()/(2*dy())) );

            eval(vy())=eval((v(i+1, j+1) +
                             v(i+1))/2.-
                            ((pow<2>(v(i+1,j+1))/h(i+1,j+1)+pow<2>(h(i+1,j+1))*g()/2.)  -
                             (pow<2>(v(i+1))/h(i+1) +
                              pow<2>(h(i+1))*(g()/2.)
                                 ))*(dt()/(2.*dy())));

        }
    };

    struct final_step        : public functor_traits {

        using xrange=range<0,-3,0,-3>;
        using xrange_subdomain=range<1,1,1,1>;

        typedef arg_type<0, range<0,0,0,0>, 5> tmpx;
        typedef arg_type<1, range<0,0,0,0>, 5> tmpy;
        typedef arg_type<2,range<0, 0, 0, 0>, 5> sol;
        typedef boost::mpl::vector<tmpx, tmpy, sol> arg_list;
        static uint_t current_time;

        //########## FINAL STEP #############
        //data dependencies with the previous parts
        //notation: alias<tmp, comp, step>::set<0, 0>() is ==> tmp(comp(0), step(0)).
        //Using a strategy to define some arguments beforehand

        template <typename Evaluation>
        GT_FUNCTION
        static void Do(Evaluation const & eval, x_interval) {

            using hx=alias<tmpx, comp>::set<0>; using h=alias<sol, comp>::set<0>; using hy=alias<tmpy, comp>::set<0>;
            using ux=alias<tmpx, comp>::set<1>; using u=alias<sol, comp>::set<1>; using uy=alias<tmpy, comp>::set<1>;
            using vx=alias<tmpx, comp>::set<2>; using v=alias<sol, comp>::set<2>; using vy=alias<tmpy, comp>::set<2>;

            eval(sol()) =
                eval(sol()-
                     (ux(j-1) - ux(i-1, j-1))*(dt()/dx())
                     -
                     (vy(i-1) - vy(i-1, j-1))*(dt()/dy())
                    );

            eval(sol(comp(1))) =
                eval(sol(comp(1)) -
                     (pow<2>(ux(j-1))                / hx(j-1)      + hx(j-1)*hx(j-1)*((g()/2.))                 -
                      (pow<2>(ux(i-1,j-1))            / hx(i-1, j-1) +pow<2>(hx(i-1,j-1) )*((g()/2.))))*((dt()/dx())) -
                     (vy(i-1)*uy(i-1)          / hy(i-1)                                                   -
                      vy(i-1, j-1)*uy(i-1,j-1) / hy(i-1, j-1)) *(dt()/dy()));

            eval(sol(comp(2))) =
                eval(sol(comp(2)) -
                     (ux(j-1)    *vx(j-1)       /hx(j-1) -
                      (ux(i-1,j-1)*vx(i-1, j-1)) /hx(i-1, j-1))*((dt()/dx()))-
                     (pow<2>(vy(i-1))                /hy(i-1)      +pow<2>(hy(i-1)     )*((g()/2.)) -
                      (pow<2>(vy(i-1, j-1))           /hy(i-1, j-1) +pow<2>(hy(i-1, j-1))*((g()/2.))   ))*((dt()/dy())));

        }

    };


    uint_t final_step::current_time=0;

/*
 * The following operators and structs are for debugging only
 */
    std::ostream& operator<<(std::ostream& s, first_step_x const) {
        return s << "initial step 1: ";
        // initiali_step.to_string();
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

    extern char const s1[]="hello ";
    extern char const s2[]="world ";

    bool test(uint_t x, uint_t y, uint_t z, uint_t t) {
#ifndef __CUDACC__
        //testing the static printing
        typedef string_c<print, s1, s2, s1, s1 > s;
        s::apply();
#endif

        uint_t d1 = x;
        uint_t d2 = y;
        uint_t d3 = z;

#ifdef CUDA_EXAMPLE
#define BACKEND backend<Cuda, Naive >
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
        typedef field<tmp_storage_type, 1, 1, 1>::type tmp_type;
        typedef sol_type::original_storage::pointer_type ptr;

        // Definition of placeholders. The order of them reflects the order the user will deal with them
        // especially the non-temporary ones, in the construction of the domain
        typedef arg<0, tmp_type > p_tmpx;
        typedef arg<1, tmp_type > p_tmpy;
        typedef arg<2, sol_type > p_sol;
        typedef boost::mpl::vector<p_tmpx, p_tmpy, p_sol> arg_type_list;
        typedef sol_type::original_storage::pointer_type pointer_type;

        gridtools::array<int, 3> dimensions(0,0,0);
        MPI_3D_process_grid_t<3>::dims_create(PROCS, 2, dimensions);
        dimensions[2]=1;

        typedef gridtools::halo_exchange_dynamic_ut<gridtools::layout_map<2, 1, 0>,
                                                    gridtools::layout_map<0, 1, 2>,
                                                    pointer_type::pointee_t, MPI_3D_process_grid_t<3>,
#ifdef __CUDACC__
                                                    gridtools::gcl_gpu,
#else
                                                    gridtools::gcl_cpu,
#endif
                                                    gridtools::version_manual> pattern_type;

        pattern_type he(gridtools::boollist<3>(false,false,false), GCL_WORLD, &dimensions);

        array<ushort_t, 3> halo={2,2,0};
        typedef partitioner_trivial<cell_topology<topology::cartesian<layout_map<0,1,2> > >, pattern_type::grid_type> partitioner_t;
        partitioner_t part(he.comm(), halo);
        parallel_storage<sol_type, partitioner_t> sol(part);
        sol.setup(d1, d2, d3);

        he.add_halo<0>(sol.get_halo_gcl<0>());
        he.add_halo<1>(sol.get_halo_gcl<1>());
        he.add_halo<2>(0, 0, 0, d3 - 1, d3);

        he.setup(3);

        ptr out7(sol.size()), out8(sol.size()), out9(sol.size());
        // if(!he.comm().pid())
        //     sol.set<0,0>(out7, &bc_periodic<0,0>::droplet);//h
        // else
        //     sol.set<0,0>(out7, 1.);//h
        sol.set<0,0>(out7, &bc_periodic<0,0>::droplet);//h
        sol.set<0,1>(out8, 0.);//u
        sol.set<0,2>(out9, 0.);//v

#ifndef NDEBUG
    int pid=0;
    MPI_Comm_rank(MPI_COMM_WORLD, &pid);
    std::ofstream myfile;
    std::stringstream name;
    name<<"example"<<pid<<".txt";
    myfile.open (name.str().c_str());

    std::cout<<"INITIALIZED VALUES"<<std::endl;
    sol.print(myfile);
    std::cout<<"#####################################################"<<std::endl;

#endif
        // construction of the domain. The domain is the physical domain of the problem, with all the physical fields that are used, temporary and not
        // It must be noted that the only fields to be passed to the constructor are the non-temporary.
        // The order in which they have to be passed is the order in which they appear scanning the placeholders in order. (I don't particularly like this)
        domain_type<arg_type_list> domain
            (boost::fusion::make_vector(&sol));

        // Definition of the physical dimensions of the problem.
        // The constructor takes the horizontal plane dimensions,
        // while the vertical ones are set according the the axis property soon after
        coordinates<axis, partitioner_t> coords(part, sol);

        coords.value_list[0] = 0;
        coords.value_list[1] = d3-1;

        auto shallow_water_stencil =
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

        //the following might be runtime value
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
            shallow_water_stencil->run();

            std::vector<pointer_type::pointee_t*> vec={sol.fields()[0].get(), sol.fields()[1].get(), sol.fields()[2].get()};
            he.pack(vec);
            he.exchange();
            he.unpack(vec);

#ifndef NDEBUG
            shallow_water_stencil->finalize();
            sol.print(myfile);
#endif
        }

#ifdef NDEBUG
        shallow_water_stencil->finalize();
#else
        myfile.close();
#endif

        he.wait();

        GCL_Finalize();

        return true;

    }

}//namespace shallow_water
