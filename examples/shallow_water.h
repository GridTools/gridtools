#pragma once

#include <gridtools.h>
#include <common/halo_descriptor.h>
#include <boost/lambda/bind.hpp>
#include <boost/lambda/construct.hpp>
#include <boost/fusion/include/make_vector.hpp>

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

/*
  @file
  @brief This file shows an implementation of the "shallow water" stencil, with periodic boundary conditions
  It defines
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
    typedef gridtools::interval<level<0,-1>, level<1,-1> > x_interval;
    typedef gridtools::interval<level<0,-2>, level<1,1> > axis;

/**@brief This traits class defined the necessary typesand functions used by all the functors defining the shallow water model*/
    struct functor_traits{
//#if  !((defined(__GNUC__)) && (__GNUC__ < 4) || (__GNUC__ == 4 && __GNUC_MINOR__ < 9))
        using tmp=arg_extend<arg_type<0, range<-1, 1, -1, 1> >, 2>::type ;
        using sol=arg_extend<arg_type<1, range<-1, 1, -1, 1> >, 2>::type ;
        using arg_list=boost::mpl::vector<tmp, sol> ;
        using step=Dimension<3> ;
        using comp=Dimension<4>;
//#endif

	/**@brief space discretization step in direction i */
	GT_FUNCTION
        static float_type dx(){return 1e-2;}
	/**@brief space discretization step in direction j */
	GT_FUNCTION
        static float_type dy(){return 1e-2;}
	/**@brief time discretization step */
	GT_FUNCTION
        static float_type dt(){return 1e-3;}
	/**@brief gravity acceleration */
	GT_FUNCTION
        static float_type g(){return 9.81;}
    };

    template<uint_t Component=0, uint_t Snapshot=0>
    struct bc_periodic : functor_traits {
        // periodic boundary conditions in I
        template <sign I, sign K, typename DataField0>
        GT_FUNCTION
        void operator()(direction<I, minus_, K, typename boost::enable_if_c<I!=minus_>::type>,
                        DataField0 & data_field0,
                        uint_t i, uint_t j, uint_t k) const {
	    // TODO use placeholders here instead of the storage
	    data_field0.template get<Component, Snapshot>()[data_field0._index(i,j,k)] = data_field0.template get<Component, Snapshot>()[data_field0._index(i,data_field0.template dims<1>()-1-j,k)];
        }

        // periodic boundary conditions in J
        template <sign J, sign K, typename DataField0>
        GT_FUNCTION
        void operator()(direction<minus_, J, K>,
                        DataField0 & data_field0,
                        uint_t i, uint_t j, uint_t k) const {
	    // TODO use placeholders here instead of the storage
	    data_field0.template get<Component, Snapshot>()[data_field0._index(i,j,k)] = data_field0.template get<Component, Snapshot>()[data_field0._index(data_field0.template dims<0>()-1-i,j,k)];
        }

	// default: do nothing
        template <sign I, sign J, sign K, typename P, typename DataField0>
        GT_FUNCTION
        void operator()(direction<I, J, K, P>,
                        DataField0 & data_field0,
                        uint_t i, uint_t j, uint_t k) const {
        }

	static constexpr float_type height=3.;
	GT_FUNCTION
    	static float_type droplet(uint_t const& i, uint_t const& j, uint_t const& k){
    	    return height * std::exp(-500*((i*dx())*((i*dx())+(j*dy())*(j*dy()))));
       }

};

    // struct bc : functor_traits{
    //     // periodic boundary conditions in K
    //     typedef arg_type<0> bc1;
    //     typedef arg_type<0> bc2;
    // 	static const float_type height=3.;

    // 	GT_FUNCTION
    // 	static float_type droplet(uint_t const& i, uint_t const& j, uint_t const& k){
    // 	    return height * std::exp(-5*((i*dx())*((i*dx())+(j*dy())*(j*dy())));
    // 	}

    //     template < typename Evaluation>
    //     GT_FUNCTION
    // 	static void Do(Evaluation const & eval, x_interval) {
    // 	    {
    // 		bc1() = bc2;
    // 	    }
    // };

    // struct bc_vertical{
    //     // periodic boundary conditions in K
    //     template < sign K, typename DataField0, typename DataField1>
    //     GT_FUNCTION
    //     void operator()(direction<zero_, zero_, K>,
    //                     DataField0 & data_field0, DataField1 const & data_field1,
    //                     uint_t i, uint_t j, uint_t k) const {
    //         data_field0(i,j,k) = data_field1(i,j,k);
    //     }
    // };

// These are the stencil operators that compose the multistage stencil in this test
    struct initial_step        : public functor_traits {
        /**GCC 4.8.2  bug: inheriting the 'using' aliases (or replacing the typedefs below with the 'using' syntax) from the base class produces an internal compiler error (segfault).
           The compilation runs fine without warnings with GCC >= 4.9 and Clang*/
#if  (defined(__GNUC__)) && (__GNUC__ < 4) || (__GNUC__ == 4 && __GNUC_MINOR__ < 9)
        //shielding the base class aliases
        typedef arg_extend<arg_type<0, range<-1, 1, -1, 1> >, 2>::type tmp;
        typedef arg_extend<arg_type<1, range<-1, 1, -1, 1> >, 2>::type sol;
        typedef boost::mpl::vector<tmp, sol> arg_list;
        typedef Dimension<3> step;
        typedef Dimension<4> comp;
#endif
        /* static const auto expression=in(1,0,0)-out(); */

        template<typename Evaluation, typename ComponentU, typename DimensionX, typename DimensionY>
        GT_FUNCTION
        static float_type /*&&*/ half_step(Evaluation const& eval, ComponentU&& U, DimensionX&& d1, DimensionY&& d2, float_type const& delta)
            {
                return /*std::move*/(eval(sol(d1,d2) +sol(d2)/2. -
                                      (sol(U,d2,d1) - sol(U,d2))*(dt()/(2*delta))));
            }

        template<typename Evaluation, typename ComponentU, typename DimensionX, typename DimensionY>
        GT_FUNCTION
        static float_type /*&&*/ half_step_u(Evaluation const& eval, ComponentU&& U, DimensionX&& d1, DimensionY&& d2, float_type const& delta)
            {
                return /*std::move*/(eval((sol(U, d1, d2) +
					   sol(U, d2)/2. -
					   (pow<2>(sol(U,d1,d2))/sol(d1,d2)+pow<2>(sol(d1,d2))*g()/2. -
					    pow<2>(sol(U, d2))/sol(d2) +
					    pow<2>(sol(d2))*(g()/2.)))*(dt()/(2.*delta))) );
            }

        template<typename Evaluation, typename ComponentU, typename ComponentV, typename DimensionX, typename DimensionY>
        GT_FUNCTION
        static float_type/*&&*/ half_step_v(Evaluation const& eval, ComponentU&& U, ComponentV&& V, DimensionX&& d1, DimensionY&& d2, float_type const& delta)
            {
                return /*std::move*/(eval( sol(V,d1,d2) +
					   sol(V,d1)/2. -
					   (sol(U,d1,d2)*sol(V,d1,d2)/sol(d1,d2) -
					    sol(U,d2)*sol(V,d2)/sol(d2))*(dt()/(2*delta)) ) );
            }


        template <typename Evaluation>
        GT_FUNCTION
        static void Do(Evaluation const & eval, x_interval) {
            eval(tmp()       )=half_step  (eval, comp(1), x(1), y(1), dx());
	    eval(tmp(comp(1)))=half_step_u(eval, comp(1), x(1), y(1), dx());
	    eval(tmp(comp(2)))=half_step_v(eval, comp(1), comp(2), x(1), y(1), dx());
	    eval(tmp(comp(0), step(1)))=half_step  (eval, comp(2), y(1), x(1), dy());
	    eval(tmp(comp(1), step(1)))=half_step_v(eval, comp(2), comp(1), y(1), x(1), dy());
	    eval(tmp(comp(2), step(1)))=half_step_u(eval, comp(2), y(1), x(1), dy());
        }

    // 	void to_string(){
    // 	    (sol(V,d1,d2) +
    // 	     sol(V,d1)/2. -
    // 	     (sol(U,d1,d2)*sol(V,d1,d2)/sol(d1,d2) -
    // 	      sol(U,d2)*sol(V,d2)/sol(d2))*(dt()/(2*delta)) )).to_string();
    // }
    };

    struct final_step        : public functor_traits {
#if  (defined(__GNUC__)) && (__GNUC__ < 4) || (__GNUC__ == 4 && __GNUC_MINOR__ < 9)
        typedef arg_extend<arg_type<0, range<-1, 1, -1, 1> >, 2>::type tmp;
        typedef arg_extend<arg_type<1, range<-1, 1, -1, 1> >, 2>::type sol;
        typedef boost::mpl::vector<tmp, sol> arg_list;
        typedef Dimension<3> step;
        typedef Dimension<4> comp;
#endif
	static uint_t current_time;

        template <typename Evaluation>
        GT_FUNCTION
        static void Do(Evaluation const & eval, x_interval) {
            //########## FINAL STEP #############
            //data dependencies with the previous parts
            //notation: alias<tmp, comp, step>(0, 0) is ==> tmp(comp(0), step(0)).
            //Using a strategy to define some arguments beforehand

            x::Index i;
            y::Index j;
#ifdef __CUDACC__
            comp::Index c;
            step::Index s;

            eval(sol()) = eval(sol()-
                               (tmp(c+1, i-1) - tmp(c+1, i-1, j-1))*(dt()/dx())-
	    		       (tmp(c+2, s+1, i/**/-1) - tmp(c+2, s+1, i-1, j-1))*(dt()/dy())/**/);

            eval(sol(comp(1))) = eval(sol(c+1)   -
				      (pow<2>(tmp(c+1, j-1))                / tmp(j-1)     + tmp(j-1)*tmp(j-1)*((g()/2.))                 -
				       (pow<2>(tmp(c+1,i-1,j-1))            / tmp(i-1, j-1) +pow<2>(tmp(i-1,j-1) )*((g()/2.))))*((dt()/dx())) -
				      (tmp(c+2,s+1,i-1)*tmp(c+1,s+1,i-1)          / tmp(s+1,i-1)                                                   -
				       tmp(c+2,s+1,i-1, j-1)*tmp(c+1,s+1,i-1,j-1) / tmp(s+1,i-1, j-1))*((dt()/dy())));/**/
// + tmp(s+1,i-1, j-1)*((g()/2.)))    *((dt()/dy())));

            eval(sol(comp(2))) = eval(sol(comp(2)) -
	    			      (tmp(c+1,j-1)    *tmp(c+2,j-1)       /tmp(s+1,j-1) -
                                       (tmp(c+1,i-1,j-1)*tmp(c+2,i-1, j-1)) /tmp(i-1, j-1))*((dt()/dx()))-
                                      (pow<2>(tmp(c+2,s+1,i-1))                /tmp(s+1,i-1)      +pow<2>(tmp(s+1,i-1)     )*((g()/2.)) -
                                       pow<2>(tmp(c+2,s+1,i-1,j-1))           /tmp(s+1,i-1,j-1) +pow<2>(tmp(s+1,i-1, j-1))*((g()/2.))   )*((dt()/dy())));
#else

	    auto hx=alias<tmp, comp, step>(0, 0); auto hy=alias<tmp, comp, step>(0, 1);
            auto ux=alias<tmp, comp, step>(1, 0); auto uy=alias<tmp, comp, step>(1, 1);
            auto vx=alias<tmp, comp, step>(2, 0); auto vy=alias<tmp, comp, step>(2, 1);

            eval(sol()) = eval(sol()-
                               (ux(i-1) - ux(i-1, j-1))*(dt()/dx())-
                               (vy(i-1) - vy(i-1, j-1))*(dt()/dy()));

            eval(sol(comp(1))) = eval(sol(comp(1)) -
                                       (pow<2>(ux(j-1))                / hx(j-1)      + hx(j-1)*hx(j-1)*((g()/2.))                 -
	    			       (pow<2>(ux(i-1,j-1))            / hx(i-1, j-1) +pow<2>(hx(i-1,j-1) )*((g()/2.))))*((dt()/dx())) -
                                              (vy(i-1)*uy(i-1)          / hy(i-1)                                                   -
                                               vy(i-1, j-1)*uy(i-1,j-1) / hy(i-1, j-1)) *(dt()/dy()));

            eval(sol(comp(2))) = eval(sol(comp(2)) -
                                       (ux(j-1)    *vx(j-1)       /hy(j-1) -
                                       (ux(i-1,j-1)*vx(i-1, j-1)) /hx(i-1, j-1))*((dt()/dx()))-
                                      (pow<2>(vy(i-1))                /hy(i-1)      +pow<2>(hy(i-1)     )*((g()/2.)) -
                                       pow<2>(vy(i-1, j-1))           /hy(i-1, j-1) +pow<2>(hy(i-1, j-1))*((g()/2.))   )*((dt()/dy())));
#endif

    	}

    };

    uint_t final_step::current_time=0;

/*
 * The following operators and structs are for debugging only
 */
    std::ostream& operator<<(std::ostream& s, initial_step const) {
        return s << "initial step: ";
	// initiali_step.to_string();
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
            typedef gridtools::layout_map<2,1,0> layout_t;
            typedef gridtools::BACKEND::storage_type<float_type, layout_t >::type storage_type;

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
	    typedef extend<storage_type::basic_type, 1, 1, 1>::type tmp_type;
            typedef extend<storage_type::basic_type, 0, 0, 0>::type sol_type;
#endif
	    typedef tmp_type::original_storage::pointer_type ptr;

            // Definition of placeholders. The order of them reflects the order the user will deal with them
            // especially the non-temporary ones, in the construction of the domain
            typedef arg<0, tmp_type > p_tmp;
            typedef arg<1, sol_type > p_sol;
            typedef boost::mpl::vector<p_tmp, p_sol> arg_type_list;


            // // Definition of the actual data fields that are used for input/output
            tmp_type tmp(d1,d2,d3);
            ptr out1(tmp.size()), out2(tmp.size()), out3(tmp.size()), out4(tmp.size()), out5(tmp.size()), out6(tmp.size());

            sol_type sol(d1,d2,d3);
            ptr out7(sol.size()), out8(sol.size()), out9(sol.size());

	    tmp.set<0,0>(out1, 0.);
	    tmp.set<1,0>(out2, 0.);
	    tmp.set<2,0>(out3, 0.);
	    tmp.set<0,1>(out4, 0.);
	    tmp.set<1,1>(out5, 0.);
	    tmp.set<2,1>(out6, 0.);

            sol.set<0>(out7, &bc_periodic<0,0>::droplet);//h
            sol.set<1>(out8, &bc_periodic<0,1>::droplet);//u
            sol.set<2>(out9, 0.);//v

            // sol.push_front<3>(out9, [](uint_t i, uint_t j, uint_t k) ->float_type {return 2.0*exp (-5*(i^2+j^2));});//h

            // construction of the domain. The domain is the physical domain of the problem, with all the physical fields that are used, temporary and not
            // It must be noted that the only fields to be passed to the constructor are the non-temporary.
            // The order in which they have to be passed is the order in which they appear scanning the placeholders in order. (I don't particularly like this)
            gridtools::domain_type<arg_type_list> domain
                (boost::fusion::make_vector(&tmp, &sol));

            // Definition of the physical dimensions of the problem.
            // The constructor takes the horizontal plane dimensions,
            // while the vertical ones are set according the the axis property soon after
            // gridtools::coordinates<axis> coords(2,d1-2,2,d2-2);
            uint_t di[5] = {2, 2, 2, d1-2, d1-2};
            uint_t dj[5] = {2, 2, 2, d2-2, d2-2};

            gridtools::coordinates<axis> coords(di, dj);
            coords.value_list[0] = 0;
            coords.value_list[1] = d3-1;

// ///////////////////BOUNDARY CONDITIONS (TEST)///////////////////////
//             uint_t d0[5] = {0, 0, 0, 0, 0};
//             uint_t d_span_x[5] = {0, 0, 0, d1-1, d1};
//             uint_t d_span_y[5] = {0, 0, 0, d2-1, d2};

//             typedef gridtools::layout_map<-1,-1,0> layout_x;//2D storage
// 	    gridtools::coordinates<axis> coords_bc_x(d_span_x, d0);
// 	    coords.value_list[0] = 0;//only on k=0 (top)
// 	    coords.value_list[1] = 0;

// 	    typedef gridtools::layout_map<-1,0,-1> layout_y;//2D storage
// 	    gridtools::coordinates<axis> coords_bc_y(d0, d_span_y);
// 	    coords.value_list[0] = 0;//only on k=0 (top)
// 	    coords.value_list[1] = 0;

// 	    typedef gridtools::BACKEND::storage_type<float_type, layout_x >::type storage_bc_x;
// 	    typedef gridtools::BACKEND::storage_type<float_type, layout_y >::type storage_bc_y;

// 	    typedef arg<0, storage_bc_x > p_bc_x;
// 	    typedef arg<0, storage_bc_y > p_bc_y;
//             typedef boost::mpl::vector<p_bc_x> arg_type_list;
//             gridtools::domain_type<arg_type_list> domain
//                 (boost::fusion::make_vector(&tmp, &sol));

// 	    auto bc_x =
//                 gridtools::make_computation<gridtools::BACKEND, layout_x>
//                 (
//                     gridtools::make_mss // mss_descriptor
//                     (
//                         execute<forward>(),
//                         gridtools::make_esf<bc>(p_sol(), p_sol()) ,
//                         ),
//                     domain, coords_bc_x
//                     );

// 	    auto bc_y =
//                 gridtools::make_computation<gridtools::BACKEND, layout_y>
//                 (
//                     gridtools::make_mss // mss_descriptor
//                     (
//                         execute<forward>(),
//                         gridtools::make_esf<bc>(p_sol(), p_sol()) ,
//                         ),
//                     domain, coords_bc_y
//                     );
// ///////////////////END BOUNDARY CONDITIONS///////////////////////

            auto shallow_water_stencil =
                gridtools::make_computation<gridtools::BACKEND, layout_t>
                (
                    gridtools::make_mss // mss_descriptor
                    (
                        execute<forward>(),
                        gridtools::make_esf<initial_step>(p_tmp(), p_sol() ) ,
                        gridtools::make_esf<final_step>(p_tmp(), p_sol() )
                        ),
                    domain, coords
                    );

	    // bc_x->ready();
	    // bc_y->ready();
            shallow_water_stencil->ready();

	    // bc_x->steady();
	    // bc_y->steady();
            shallow_water_stencil->steady();

            gridtools::array<gridtools::halo_descriptor, 3> halos;
            halos[0] = gridtools::halo_descriptor(1,1,1,d1-2,d1);
            halos[1] = gridtools::halo_descriptor(1,1,1,d2-2,d2);
            halos[2] = gridtools::halo_descriptor(0,0,0,0,0);

	    //the following might be runtime value
	    uint_t total_time=3;

	    // bc_x->run();
	    // bc_y->run();
	    for (;final_step::current_time <= total_time; ++final_step::current_time)
	    {
#ifdef CUDA_EXAMPLE
		// TODO: use placeholders here instead of the storage
		/*                                 component,snapshot */
		gridtools::boundary_apply_gpu< bc_periodic<0,0> >(halos, bc_periodic<0,0>()).apply(sol);
		gridtools::boundary_apply_gpu< bc_periodic<1,0> >(halos, bc_periodic<1,0>()).apply(sol);
#else
		// TODO: use placeholders here instead of the storage
		/*                             component,snapshot */
		gridtools::boundary_apply< bc_periodic<0,0> >(halos, bc_periodic<0,0>()).apply(sol);
		gridtools::boundary_apply< bc_periodic<1,0> >(halos, bc_periodic<1,0>()).apply(sol);
#endif
		shallow_water_stencil->run();
	    }

            shallow_water_stencil->finalize();

            // tmp.print();
	    sol.print();
        }
        return true;

    }

}//namespace shallow_water
