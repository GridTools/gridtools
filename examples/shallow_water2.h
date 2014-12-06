
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
  @brief This file shows an implementation of the "shallow water" stencil
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

namespace shallow_water2{
// This is the definition of the special regions in the "vertical" direction
    typedef gridtools::interval<level<0,-1>, level<1,-1> > x_interval;
    typedef gridtools::interval<level<0,-2>, level<1,1> > axis;

//     struct bc_reflecting {
//         // reflective boundary conditions in I and J
//         template <sign I, sign J, typename DataField0, typename DataField1>
//         GT_FUNCTION
//         void operator()(direction<I, J, zero_>,
//                         DataField0 & data_field0, DataField1 const & data_field1,
//                         uint_t i, uint_t j, uint_t k) const {
// // TODO use placeholders here instead of the storage
//             data_field0(i,j,k) = data_field1(i,j,k);
//         }
//     };

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
    struct initial_step  {
        /**GCC 4.8.2  bug: inheriting the 'using' aliases (or replacing the typedefs below with the 'using' syntax) from the base class produces an internal compiler error (segfault).
           The compilation runs fine without warnings with GCC >= 4.9 and Clang*/
        //shielding the base class aliases
        typedef arg_type<0, range<-1, 1, -1, 1> > hx;
        typedef arg_type<1, range<-1, 1, -1, 1> > ux;
        typedef arg_type<2, range<-1, 1, -1, 1> > vx;
        typedef arg_type<3, range<-1, 1, -1, 1> > hy;
        typedef arg_type<4, range<-1, 1, -1, 1> > uy;
        typedef arg_type<5, range<-1, 1, -1, 1> > vy;
        typedef arg_type<6, range<-1, 1, -1, 1> > h;
        typedef arg_type<7, range<-1, 1, -1, 1> > u;
        typedef arg_type<8, range<-1, 1, -1, 1> > v;
        typedef boost::mpl::vector<hx, ux, vx, hy, uy, vy, h, u, v> arg_list;

        static float_type dx(){return 1e-2;}
        static float_type dy(){return 1e-2;}
        static float_type dt(){return 1e-3;}
        static float_type g(){return 9.81;}


        template <typename Evaluation>
        GT_FUNCTION
        static void Do(Evaluation const & eval, x_interval) {
            x::Index i;
            y::Index j;

            eval(hx()       )=  (eval(h(i+1,j+1) +h(j+1)/2. -
                                      (u(i+1,j+1) - u(j+1))*(dt()/(2*dx()))));

            eval(ux())=(eval((u(i+1, j+1) +
			      u(j+1)/2. -
			      (pow<2>(u(i+1,j+1))/h(i+1,j+1)+pow<2>(h(i+1,j+1))*g()/2.)*(dt()/(2.*dx())) -
			      pow<2>(u(j+1))/h(j+1) -
			      pow<2>(h(j+1))*pow<2>(g()/2.))) );

            eval(vx())=(eval(( v(i+1,j+1) +
				       v(i+1)/2. -
				       u(i+1,j+1)*v(i+1,j+1)/h(i+1,j+1)*(dt()/(2*dx())) -
				       u(j+1)*v(j+1)/h(j+1))));

            eval(hy()       )=  (eval(h(j+1,i+1) +h(j+1)/2. -
                                      (v(j+1,i+1) - v(i+1))*(dt()/(2*dy()))));

            eval(uy())=(eval(( u(j+1,i+1) +
				       u(j+1)/2. -
				       v(j+1,i+1)*v(j+1,i+1)/h(j+1,i+1)*(dt()/(2*dy())) -
				       v(i+1)*u(i+1)/h(i+1))));

            eval(vy())=(eval((v(j+1, i+1) +
			      v(i+1)/2. -
			      (pow<2>(v(j+1,i+1))/h(j+1,i+1)+pow<2>(h(j+1,i+1))*g()/2.)*(dt()/(2.*dy())) -
			      pow<2>(v(i+1))/h(i+1) -
			      pow<2>(h(i+1))*pow<2>(g()/2.))) );

        }
    };

struct final_step {

    typedef arg_type<0, range<-1, 1, -1, 1> > hx;
	    typedef arg_type<1, range<-1, 1, -1, 1> > ux;
	    typedef arg_type<2, range<-1, 1, -1, 1> > vx;
	    typedef arg_type<3, range<-1, 1, -1, 1> > hy;
	    typedef arg_type<4, range<-1, 1, -1, 1> > uy;
	    typedef arg_type<5, range<-1, 1, -1, 1> > vy;
	    typedef arg_type<6, range<-1, 1, -1, 1> > h;
	    typedef arg_type<7, range<-1, 1, -1, 1> > u;
	    typedef arg_type<8, range<-1, 1, -1, 1> > v;
	    typedef boost::mpl::vector<hx, ux, vx, hy, uy, vy, h, u, v> arg_list;

	    static float_type dx(){return 1e-2;}
	    static float_type dy(){return 1e-2;}
	    static float_type dt(){return 1e-3;}
	    static float_type g(){return 9.81;}

        template <typename Evaluation>
        GT_FUNCTION
        static void Do(Evaluation const & eval, x_interval) {
            //########## FINAL STEP #############
            //data dependencies with the previous parts
            //notation: alias<tmp, comp, step>(0, 0) is ==> tmp(comp(0), step(0)).
            //Using a strategy to define some arguments beforehand

            x::Index i;
            y::Index j;

            eval(h()) = eval(h()-
                               (ux(i-1) - ux(i-1, j-1))*(dt()/dx())-
                               vy(j-1) - vy(i-1, j-1)*(dt()/dy()));

            eval(u()) =  eval(u() -
                                       (pow<2>(ux(j-1))                / hx(j-1)      + hx(j-1)*hx(j-1)*((g()/2.))                 -
	    			       (pow<2>(ux(i-1,j-1))            / hx(i-1, j-1) +pow<2>(hx(i-1,j-1) )*((g()/2.))))*((dt()/dx())) -
                                              (vy(i-1)*uy(i-1)          / hy(i-1)                                                   -
                                               vy(i-1, j-1)*uy(i-1,j-1) / hy(i-1, j-1) + hy(i-1, j-1)*((g()/2.)))    *((dt()/dy())));

            eval(v()) = eval(v() -
			     (ux(j-1)    *vx(j-1)       /hy(j-1) -
			      (ux(i-1,j-1)*vx(i-1, j-1)) /hx(i-1, j-1))*((dt()/dx()))-
			     (pow<2>(vy(i-1))                /hy(i-1)      +pow<2>(hy(i-1)     )*((g()/2.)) -
			      pow<2>(vy(i-1, j-1))           /hy(i-1, j-1) +pow<2>(hy(i-1, j-1))*((g()/2.))   )*((dt()/dy())));

    	}

    };

/*
 * The following operators and structs are for debugging only
 */
    std::ostream& operator<<(std::ostream& s, initial_step const) {
        return s << "initial step";
    }


/*
 * The following operators and structs are for debugging only
 */
    std::ostream& operator<<(std::ostream& s, final_step const) {
        return s << "final step";
    }


    void handle_error(int_t)
    {std::cout<<"error"<<std::endl;}

    bool test(uint_t x, uint_t y, uint_t z) {
        {

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

            // Definition of placeholders. The order of them reflect the order the user will deal with them
            // especially the non-temporary ones, in the construction of the domain

	    storage_type hx(d1,d2,d3),ux(d1,d2,d3),vx(d1,d2,d3),hy(d1,d2,d3),uy(d1,d2,d3),vy(d1,d2,d3),h(d1,d2,d3),u(d1,d2,d3),v(d1,d2,d3);
            typedef arg<0, storage_type > p_hx;
            typedef arg<1, storage_type > p_ux;
            typedef arg<2, storage_type > p_vx;
            typedef arg<3, storage_type > p_hy;
            typedef arg<4, storage_type > p_uy;
            typedef arg<5, storage_type > p_vy;
            typedef arg<6, storage_type > p_h;
            typedef arg<7, storage_type > p_u;
            typedef arg<8, storage_type > p_v;
            typedef boost::mpl::vector<p_hx, p_ux, p_vx, p_hy, p_uy, p_vy, p_h, p_u, p_v> arg_type_list;

            // construction of the domain. The domain is the physical domain of the problem, with all the physical fields that are used, temporary and not
            // It must be noted that the only fields to be passed to the constructor are the non-temporary.
            // The order in which they have to be passed is the order in which they appear scanning the placeholders in order. (I don't particularly like this)
            gridtools::domain_type<arg_type_list> domain
                (boost::fusion::make_vector(&hx, &ux, &vx, &hy, &uy, &vy, &h, &u, &v));

            // Definition of the physical dimensions of the problem.
            // The constructor takes the horizontal plane dimensions,
            // while the vertical ones are set according the the axis property soon after
            // gridtools::coordinates<axis> coords(2,d1-2,2,d2-2);
            uint_t di[5] = {2, 2, 2, d1-2, d1-2};
            uint_t dj[5] = {2, 2, 2, d2-2, d2-2};

            gridtools::coordinates<axis> coords(di, dj);
            coords.value_list[0] = 0;
            coords.value_list[1] = d3-1;

            auto shallow_water_stencil =
                gridtools::make_computation<gridtools::BACKEND, layout_t>
                (
                    gridtools::make_mss // mss_descriptor
                    (
                        execute<forward>(),
                        gridtools::make_esf<initial_step>(p_hx(), p_ux(), p_vx(), p_hy(), p_uy(), p_vy(), p_h(), p_u(), p_v() ) ,
                        gridtools::make_esf<final_step>(p_hx(), p_ux(), p_vx(), p_hy(), p_uy(), p_vy(), p_h(), p_u(), p_v() )
                        ),
                    domain, coords
                    );

            shallow_water_stencil->ready();

            shallow_water_stencil->steady();

            gridtools::array<gridtools::halo_descriptor, 2> halos;
            halos[0] = gridtools::halo_descriptor(1,1,1,d1-2,d1);
            halos[1] = gridtools::halo_descriptor(1,1,1,d2-2,d2);
            // TODO: use placeholders here instead of the storage
            //gridtools::boundary_apply< bc_reflecting<uint_t> >(halos, bc_reflecting<uint_t>()).apply(p_sol(), p_sol());

            shallow_water_stencil->run();

            shallow_water_stencil->finalize();

            h.print();
        }
        return true;

    }

}//namespace shallow_water
