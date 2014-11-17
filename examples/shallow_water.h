
#pragma once
#ifdef CXX11_ENABLED

#include <gridtools.h>
#ifdef CUDA_EXAMPLE
#include <stencil-composition/backend_cuda.h>
#else
#include <stencil-composition/backend_host.h>
#endif

/*
  This file shows an implementation of the "shallow water" stencil, similar to the one used in COSMO
 */

using gridtools::level;
using gridtools::arg_type;
using gridtools::range;
using gridtools::arg;

using namespace gridtools;
using namespace enumtype;
using namespace expressions;
namespace shallow_water{
// This is the definition of the special regions in the "vertical" direction
    typedef gridtools::interval<level<0,-1>, level<1,-1> > x_result;
    typedef gridtools::interval<level<0,-2>, level<1,3> > axis;

    // struct bc_reflecting{
    //     // relative coordinates
    //     template <sign I, sign K, typename DataField0, typename DataField1>
    //     GT_FUNCTION
    //     void operator()(direction<I, minus_, K>,
    //                     DataField0 & data_field0, DataField1 const & data_field1,
    //                     uint_t i, uint_t j, uint_t k) const {
    //         // std::cout << "Implementation going A-A" << std::endl;
    //         data_field0(i,j,k) = data_field_1(ijk);
    //         //printf("*m* value = %d\n", value);
    //     }
    // };

    //TODO:
    //* I might want to be able to specity different ranges for different snapshots?
    //* I want a simplified notation, using currying and () instead of arg_type constructor
    //* The storage should take ownership of the fields it contains (hybrid_pointer smart)

        static constexpr float_type dx=1e-2;
        static constexpr float_type dy=1e-2;
        static constexpr float_type dt=1e-3;
        static constexpr float_type g=9.8;
    // These are the stencil operators that compose the multistage stencil in this test
    struct functor_sw {
        /**Defining 3 fields with the same size and memory access pattern, for the components of the velocity field, with ranges +1/-1 in the i/j components.
           2 extra dimensions: one for the time (not really necessary in this simple case), and another for the velocity component.*/
        typedef const  arg_extend<arg_type<0, range<-1, 1, -1, 1> >, 2>::type sol;
        typedef boost::mpl::vector<sol> arg_list;
        typedef Dimension<3> time;
        typedef Dimension<4> comp;

        template<typename DimensionX, typename DimensionY>
        GT_FUNCTION
        static auto half_step(DimensionX d1, DimensionY d2, float_type const& delta)
            {
                return sol(d1,d2)/*+sol(d2)/2.*/ /*-
                                               (sol(comp(1),d2,d1) - sol(comp(1),d2))*(dt/(2*delta))*/;
            }

        template<typename DimensionX, typename DimensionY>
        GT_FUNCTION
        static auto half_step_u(DimensionX d1, DimensionY d2, float_type const& delta)
            {
                return (sol(comp(1), d1, d2) /*+
                        sol(comp(1), d2)/2. -
                        (sol(comp(1),d1,d2)*sol(comp(1),d1,d2)/sol(d1,d2)+sol(d1,d2)*sol(d1,d2)*g/2.)*(dt/(2.*delta)) -
                        sol(comp(1), d2)*sol(comp(1), d2)/sol(d2) -
                        sol(d2)*sol(d2)*(g/2.)*/);

            }

        template<typename DimensionX, typename DimensionY>
        GT_FUNCTION
        static auto half_step_v(DimensionX d1, DimensionY d2, float_type const& delta)
            {
                return ( sol(comp(2),d1,d2)/*/2. -
                         sol(comp(1),d1,d2)*sol(comp(2),d1,d2)/sol(d1,d2)*(dt/(2*delta)) -
                         sol(comp(1),d2)*sol(comp(2),d2)/sol(d2)*/);
            }

        template <typename Evaluation>
        GT_FUNCTION
        static void Do(Evaluation const & eval, x_result) {
//                                      eval(sol(x(+1),y(+1))+sol(y(+1))/2. -
//                                      (sol(comp(1),y(+1),x(+1)) - sol(comp(1),y(+1)))*(dt/(2*dx)))

                // (sol(comp(1), x(+1), y(+1)) +
                //  sol(comp(1), y(+1))/2. -
                //  (sol(comp(1),x(+1),y(+1))*sol(comp(1),x(+1),y(+1))/sol(x(+1),y(+1))+sol(x(+1),y(+1))*sol(x(+1),y(+1))*g/2.)*(dt/(2.*dx)) -
                //  sol(comp(1), y(+1))*sol(comp(1), y(+1))/sol(y(+1)) -
                //  sol(y+1)*sol(y+1)*(g/2.));

                // ( sol(comp(2),x(+1),y(+1))/2. -
                //   sol(comp(1),x(+1),y(+1))*sol(comp(2),x(+1),y(+1))/sol(x(+1),y(+1))*(dt/(2*dx)) -
                //   sol(comp(1),y(+1))*sol(comp(2),y(+1))/sol(y(+1)));

            //########## FIRST HALF-STEP #############
            //comp(0) is the height, shallow water equation
            eval(sol(time(+1)))         =eval(half_step(x(+1), y(+1), dx));
            //comp(1) is the U component, momentum equation
            eval(sol(comp(1), time(+1)))=eval(half_step_u(x(+1), y(+1), dx));
            //comp(2) is the V component, momentum equation
            eval(sol(comp(2), time(+1)))=eval(half_step_v(x(+1), y(+1), dx));

            //########## SECOND HALF-STEP #############
            eval(sol(time(+1)))         =eval(half_step(y(+1), x(+1), dy));
            eval(sol(comp(1), time(+1)))=eval(half_step_u (y(+1), x(+1), dy));
            eval(sol(comp(2), time(+1)))=eval(half_step_v (y(+1), x(+1), dy));

        }
    };


bool test(uint_t x, uint_t y, uint_t z) {

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

    typedef gridtools::layout_map<0,1,2> layout_t;

    typedef gridtools::BACKEND::storage_type<float_type, layout_t >::type storage_type;

    /** buffer of storages for the time integration (all the components are derived in time)*/
    typedef extend_width<storage_type::basic_type, 1> extended_type;

    /** buffer of 3 storages for the different components (H, U, V), all three derived in time*/
    typedef extend_dim<extended_type, extended_type, extended_type> velocity_type;

    typedef arg<0, velocity_type > p_vel;
    typedef boost::mpl::vector<p_vel> arg_type_list;
    velocity_type vel(d1, d2, d3);

    /**populate the storage container with all the necessary storage space (6 fields in total)*/
    for (ushort_t i=0; i<5; ++i)
        vel.push_back_new();

    gridtools::domain_type<arg_type_list> domain ( (p_vel() = vel) );

    uint_t di[5] = {2, 2, 2, d1-2, d1};
    uint_t dj[5] = {2, 2, 2, d2-2, d2};

    gridtools::coordinates<axis> coords(di, dj);
    coords.value_list[0] = 0;
    coords.value_list[1] = d3-1;


// \todo simplify the following using the auto keyword from C++11
    auto model =
        gridtools::make_computation<gridtools::BACKEND, layout_t>
        (
            gridtools::make_mss // mss_descriptor
            (
                execute<forward>(),
                gridtools::make_esf<functor_sw>(p_vel())), // esf_descriptor
            domain, coords
            );

    uint_t T=10;

    model->ready();
    model->steady();

    for (ushort_t i=0; i<T; ++i){
        model->run();
        vel.advance();
    }

}
}

#endif //CXX11
