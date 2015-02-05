#pragma once

#include <gridtools.h>
#include <common/halo_descriptor.h>
#include <boost/lambda/bind.hpp>
#include <boost/lambda/construct.hpp>
#include <stencil-composition/interval.h>
#include <stencil-composition/make_computation.h>

#ifdef CUDA_EXAMPLE
#include <stencil-composition/backend_cuda.h>
#else
#include <stencil-composition/backend_host.h>
#endif

#ifdef CUDA_EXAMPLE
#include <boundary-conditions/apply_gpu.h>
#else
#include <boundary-conditions/apply.h>
#endif

#define BACKEND_BLOCK 1
/*
  @file
  @brief This file shows an implementation of the "shallow water" stencil, with periodic boundary conditions
  It is the most human readable and efficient solution among the versions implemented, but it must be compiled for the host, with Clang or GCC>=4.9, and with C++11 enabled
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
            if(i>0 && j>0 && i<4 && j<4)
                return 1.+height * std::exp(-5*(((i-1)*dx())*(((i-1)*dx()))+((j-1)*dy())*((j-1)*dy())));
            else
                return 1.;
       }

};

// These are the stencil operators that compose the multistage stencil in this test
    struct first_step_x        : public functor_traits {

        //using xrange=range<0,0,0,-1>;
        typedef arg_type<0, range<0, 0, 0, 0>, 5> tmpx;
        typedef arg_type<1, range<0, 0, 0, 0>, 5> sol;
        using arg_list=boost::mpl::vector<tmpx, sol> ;

        template <typename Evaluation>
        GT_FUNCTION
        static void Do(Evaluation const & eval, x_interval) {
        auto hx=alias<tmpx, comp>(0); auto h=alias<sol, comp>(0);
        auto ux=alias<tmpx, comp>(1); auto u=alias<sol, comp>(1);
        auto vx=alias<tmpx, comp>(2); auto v=alias<sol, comp>(2);

        eval(hx())= eval((h(i+1,j+1) +h(j+1))/2. -
                         (u(i+1,j+1) - u(j+1))*(dt()/(2*dx())));

        eval(ux())=eval(u(i+1, j+1) +
                        u(j+1)/2.-
                        ((pow<2>(u(i+1,j+1))/h(i+1,j+1)+pow<2>(h(i+1,j+1))*g()/2.)  -
                         (pow<2>(u(j+1))/h(j+1) +
                         pow<2>(h(j+1))*(g()/2.)
                             ))*(dt()/(2.*dx())));

        eval(vx())=eval( (v(i+1,j+1) +
                          v(j+1))/2. -
                         (u(i+1,j+1)*v(i+1,j+1)/h(i+1,j+1) -
                          u(j+1)*v(j+1)/h(j+1))*(dt()/(2*dx())) );
        }
    };


    struct second_step_y        : public functor_traits {

        //using xrange=range<0,-1,0,0>;
        typedef arg_type<0,range<0, 0, 0, 0>, 5> tmpy;
        typedef arg_type<1,range<0, 0, 0, 0>, 5> sol;
        using arg_list=boost::mpl::vector<tmpy, sol> ;

        template <typename Evaluation>
        GT_FUNCTION
        static void Do(Evaluation const & eval, x_interval) {

        auto hy=alias<tmpy, comp>(0); auto h=alias<sol, comp>(0);
        auto uy=alias<tmpy, comp>(1); auto u=alias<sol, comp>(1);
        auto vy=alias<tmpy, comp>(2); auto v=alias<sol, comp>(2);

        eval(hy())= eval((h(i+1,j+1) + h(i+1))/2. -
                         (v(i+1,j+1) - v(i+1))*(dt()/(2*dy())) );

        eval(uy())=eval( (u(i+1,j+1) +
                          u(i+1))/2. -
                         (v(i+1,j+1)*u(i+1,j+1)/h(i+1,j+1) -
                          v(i+1)*u(i+1)/h(i+1))*(dt()/(2*dy())) );

        eval(vy())=eval(v(i+1, j+1) +
                        v(i+1)/2.-
                        ((pow<2>(v(i+1,j+1))/h(i+1,j+1)+pow<2>(h(i+1,j+1))*g()/2.)  -
                         (pow<2>(v(i+1))/h(i+1) +
                          pow<2>(h(i+1))*(g()/2.)
                             ))*(dt()/(2.*dy())));
        }
    };

    struct final_step        : public functor_traits {

        //using xrange=range<1,0,0,0>;
        typedef arg_type<0, range<-1,0,-1,1>, 5> tmpx;
        typedef arg_type<1, range<-1,1,-1,0>, 5> tmpy;
        typedef arg_type<2,range<0, 0, 0, 0>, 5> sol;
        typedef boost::mpl::vector<tmpx, tmpy, sol> arg_list;
	static uint_t current_time;

        //########## FINAL STEP #############
        //data dependencies with the previous parts
        //notation: alias<tmp, comp, step>(0, 0) is ==> tmp(comp(0), step(0)).
        //Using a strategy to define some arguments beforehand

        // static constexpr auto hx=alias<tmpy, comp>(0); static constexpr auto hy=alias<sol, comp>(0);
        // static constexpr auto ux=alias<tmpy, comp>(1); static constexpr auto uy=alias<sol, comp>(0);
        // static constexpr auto vx=alias<tmpy, comp>(2); static constexpr auto vy=alias<sol, comp>(0);
        // typedef decltype(hx) hx_t;         typedef decltype(hy) hy_t;
        // typedef decltype(ux) ux_t;         typedef decltype(uy) uy_t;
        // typedef decltype(vx) vx_t;         typedef decltype(vy) vy_t;

        template <typename Evaluation>
        GT_FUNCTION
        static void Do(Evaluation const & eval, x_interval) {


	    auto hx=alias<tmpx, comp>(0); auto hy=alias<tmpy, comp>(0);
            auto ux=alias<tmpx, comp>(1); auto uy=alias<tmpy, comp>(1);
            auto vx=alias<tmpx, comp>(2); auto vy=alias<tmpy, comp>(2);

            eval(sol()) = eval(sol()-
                               (ux(j-1) - ux(i-1, j-1))*(dt()/dx())
                               -
                               (vy(i-1) - vy(i-1, j-1))*(dt()/dy())
                );

            eval(sol(comp(1))) =  eval(sol(comp(1)) -
                                       (pow<2>(ux(j-1))                / hx(j-1)      + hx(j-1)*hx(j-1)*((g()/2.))                 -
	    			       (pow<2>(ux(i-1,j-1))            / hx(i-1, j-1) +pow<2>(hx(i-1,j-1) )*((g()/2.))))*((dt()/dx())) -
                                              (vy(i-1)*uy(i-1)          / hy(i-1)                                                   -
                                               vy(i-1, j-1)*uy(i-1,j-1) / hy(i-1, j-1)) *(dt()/dy()));

            eval(sol(comp(2))) = eval(sol(comp(2)) -
                                       (ux(j-1)    *vx(j-1)       /hx(j-1) -
                                        (ux(i-1,j-1)*vx(i-1, j-1)) /hx(i-1, j-1))*((dt()/dx()))-
                                      (pow<2>(vy(i-1))                /hy(i-1)      +pow<2>(hy(i-1)     )*((g()/2.)) -
                                       (pow<2>(vy(i-1, j-1))           /hy(i-1, j-1) +pow<2>(hy(i-1, j-1))*((g()/2.))   ))*((dt()/dy())));
        }

    };

    // constexpr final_step::hy_t final_step::hy;
    // constexpr final_step::uy_t final_step::uy;
    // constexpr final_step::vy_t final_step::vy;
    // constexpr final_step::hx_t final_step::hx;
    // constexpr final_step::ux_t final_step::ux;
    // constexpr final_step::vx_t final_step::vx;



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

    bool test(uint_t x, uint_t y, uint_t z) {
        {
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
#ifdef __CUDACC__
//pointless and tedious syntax, temporary while thinking/waiting for an alternative like below
	    typedef base_storage<Cuda, float_type, layout_t, false ,6> base_type1;
	    typedef extend_width<base_type1, 1>  extended_type;
	    typedef extend_dim<extended_type, extended_type, extended_type>  tmp_type;

	    typedef base_storage<Cuda, float_type, layout_t, false ,3> base_type2;
	    typedef extend_width<base_type2, 0>  extended_type2;
	    typedef extend_dim<extended_type2, extended_type2, extended_type2>  sol_type;
#else
            typedef field<storage_type, 1, 1, 1>::type sol_type;
            typedef field<tmp_storage_type, 1, 1, 1>::type tmp_type;
#endif
	    typedef sol_type::original_storage::pointer_type ptr;

            // Definition of placeholders. The order of them reflects the order the user will deal with them
            // especially the non-temporary ones, in the construction of the domain
            typedef arg<0, tmp_type > p_tmpx;
            typedef arg<1, tmp_type > p_tmpy;
            typedef arg<2, sol_type > p_sol;
            typedef boost::mpl::vector<p_tmpx, p_tmpy, p_sol> arg_type_list;


            // // Definition of the actual data fields that are used for input/output
            sol_type sol(d1,d2,d3);
            ptr out7(sol.size()), out8(sol.size()), out9(sol.size());

            sol.set<0>(out7, &bc_periodic<0,0>::droplet);//h
            sol.set<1>(out8, 0.);//u
            sol.set<2>(out9, 0.);//v

            std::cout<<"INITIALIZED VALUES"<<std::endl;
            sol.print();
            std::cout<<"#####################################################"<<std::endl;

            // construction of the domain. The domain is the physical domain of the problem, with all the physical fields that are used, temporary and not
            // It must be noted that the only fields to be passed to the constructor are the non-temporary.
            // The order in which they have to be passed is the order in which they appear scanning the placeholders in order. (I don't particularly like this)
            domain_type<arg_type_list> domain
                (boost::fusion::make_vector(&sol));

            // Definition of the physical dimensions of the problem.
            // The constructor takes the horizontal plane dimensions,
            // while the vertical ones are set according the the axis property soon after
            // coordinates<axis> coords(2,d1-2,2,d2-2);
            uint_t di2[5] = {1, 0, 1, d1-3, d1};
            uint_t dj2[5] = {1, 0, 1, d2-3, d2};
            coordinates<axis> coords(di2, dj2);
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

            array<halo_descriptor, 3> halos;
            halos[0] = halo_descriptor(1,0,1,d1-1,d1);
            halos[1] = halo_descriptor(1,0,1,d2-1,d2);
            halos[2] = halo_descriptor(0,0,1,d3-1,d3);

	    //the following might be runtime value
	    uint_t total_time=1;

	    for (;final_step::current_time < total_time; ++final_step::current_time)
	    {
#ifdef CUDA_EXAMPLE
		// TODO: use placeholders here instead of the storage
		/*                                 component,snapshot */
		boundary_apply_gpu< bc_periodic<0,0> >(halos, bc_periodic<0,0>()).apply(sol);
		boundary_apply_gpu< bc_periodic<1,0> >(halos, bc_periodic<1,0>()).apply(sol);
#else
		// TODO: use placeholders here instead of the storage
		/*                             component,snapshot */
		boundary_apply< bc_periodic<0,0> >(halos, bc_periodic<0,0>()).apply(sol);
		boundary_apply< bc_periodic<1,0> >(halos, bc_periodic<1,0>()).apply(sol);
#endif
		shallow_water_stencil->run();
                sol.print();
	    }

            shallow_water_stencil->finalize();
        }
        return true;

    }

}//namespace shallow_water
