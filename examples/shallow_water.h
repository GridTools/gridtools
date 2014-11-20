#pragma once

#include <gridtools.h>
#include <common/halo_descriptor.h>
#include <boost/lambda/bind.hpp>
#include <boost/lambda/construct.hpp>

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

#ifdef CXX11_ENABLED

namespace gridtools
{

/**this struct allows the specification of SOME of the arguments before instantiating the arg_type
   it must become a language keyword (move it to arg_type.h?)
*/
template <typename Callable, typename ... Known>
struct alias{

    alias( Known&& ... dims ): m_knowns{dims.value ...} {
    };

    //operator calls the constructor of the arg_type
    template<typename ... Unknowns>
    Callable& operator()  ( Unknowns ... unknowns  )
        {return Callable(enumtype::Dimension<Known::direction> (m_knowns[Known::direction]) ... , unknowns ...);}

private:
    //store the list of offsets which are already known on an array
    int_t m_knowns [sizeof...(Known)];
};
}

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
    typedef gridtools::interval<level<0,-1>, level<1,-1> > x_result;
    typedef gridtools::interval<level<0,-2>, level<1,3> > axis;

    struct bc_reflecting{
        // reflective boundary conditions in I and J
        template <sign I, sign J, typename DataField0, typename DataField1>
        GT_FUNCTION
        void operator()(direction<I, J, zero_>,
                        DataField0 & data_field0, DataField1 const & data_field1,
                        uint_t i, uint_t j, uint_t k) const {
            data_field0(i,j,k) = data_field1(i,j,k);
        }
    };

    struct bc_vertical{
        // reflective boundary conditions in K
        template < sign K, typename DataField0, typename DataField1>
        GT_FUNCTION
        void operator()(direction<zero_, zero_, K>,
                        DataField0 & data_field0, DataField1 const & data_field1,
                        uint_t i, uint_t j, uint_t k) const {
            data_field0(i,j,k) = data_field1(i,j,k);
        }
    };

    //TODO:
    //* I might want to be able to specity different ranges for different snapshots?
    //* I want a simplified notation, using currying and () instead of arg_type constructor <-- DONE somehow with the alias
    //* The storage should take ownership of the fields it contains (hybrid_pointer smart)

    // These are the stencil operators that compose the multistage stencil in this test
    struct functor_sw {
        /**Defining 3 fields with the same size and memory access pattern, for the components of the velocity field, with ranges +1/-1 in the i/j components.
           2 extra dimensions: one for the half-step considered, and another for the velocity component.*/
        typedef const  arg_extend<arg_type<0, range<-1, 1, -1, 1> >, 2>::type tmp;
        /** Defining 3 fields with the same size and memory access pattern, for the components of the velocity field, with ranges +1/-1 in the i/j components.
           1 extra dimensions for the velocity component.*/
        typedef const  arg_extend<arg_type<1, range<-1, 1, -1, 1> >, 1>::type sol;
        typedef boost::mpl::vector<sol, tmp> arg_list;
        typedef Dimension<3> step;
        typedef Dimension<4> comp;
        static float_type dx(){return 1e-2;}
        static float_type dy(){return 1e-2;}
        static float_type dt(){return 1e-3;}
        static float_type g(){return 9.81;}

        template<typename DimensionX, typename DimensionY>
        GT_FUNCTION
        static auto half_step(DimensionX d1, DimensionY d2, float_type const& delta)
            {
                return tmp(d1,d2)+tmp(d2)/2. -
                    (tmp(comp(1),d2,d1) - tmp(comp(1),d2))*(dt()/(2*delta));
            }

        template<typename DimensionX, typename DimensionY>
        GT_FUNCTION
        static auto half_step_u(DimensionX d1, DimensionY d2, float_type const& delta)
            {
               return (tmp(comp(1), d1, d2) +
                        tmp(comp(1), d2)/2. -
                        (tmp(comp(1),d1,d2)*tmp(comp(1),d1,d2)/tmp(d1,d2)+tmp(d1,d2)*tmp(d1,d2)*g()/2.)*(dt()/(2.*delta)) -
                        tmp(comp(1), d2)*tmp(comp(1), d2)/tmp(d2) -
                        tmp(d2)*tmp(d2)*(g()/2.));
            }

        template<typename DimensionX, typename DimensionY>
        GT_FUNCTION
        static auto half_step_v(DimensionX d1, DimensionY d2, float_type const& delta)
            {
                return ( tmp(comp(2),d1,d2)/2. -
                         tmp(comp(1),d1,d2)*tmp(comp(2),d1,d2)/tmp(d1,d2)*(dt()/(2*delta)) -
                         tmp(comp(1),d2)*tmp(comp(2),d2)/tmp(d2));
            }

        template <typename Evaluation>
        GT_FUNCTION
        static void Do(Evaluation const & eval, x_result) {
            //########## FIRST HALF-STEP #############
            //comp(0) is the height, shallow water equation
            eval(sol())       =eval(half_step(x(+1), y(+1), dx()));
            //comp(1) is the U component, momentum equation
            eval(sol(comp(1)))=eval(half_step_u(x(+1), y(+1), dx()));
            //comp(2) is the V component, momentum equation
            eval(sol(comp(2)))=eval(half_step_v(x(+1), y(+1), dx()));

            //########## SECOND HALF-STEP #############
            eval(sol(step(+1)))         =eval(half_step(y(+1), x(+1), dy()));
            eval(sol(comp(1), step(+1)))=eval(half_step_u (y(+1), x(+1), dy()));
            eval(sol(comp(2), step(+1)))=eval(half_step_v (y(+1), x(+1), dy()));

            //########## FINAL STEP #############
            //data dependencies with the previous parts
            //notation: alias<tmp, comp, step>(0, 0) is ==> tmp(comp(0), step(0)).
            //Using a strategy to define some arguments beforehand
            auto hx=alias<tmp, comp, step>(0, 0); auto hy=alias<tmp, comp, step>(0, 1);
            auto ux=alias<tmp, comp, step>(1, 0); auto uy=alias<tmp, comp, step>(1, 1);
            auto vx=alias<tmp, comp, step>(2, 0); auto vy=alias<tmp, comp, step>(2, 1);

            eval(sol()) = eval(sol()-
                               (ux(y(-1)) - ux(x(-1), y(-1)))*(dt()/dx())-
                                vy(y(-1)) - vy(x(-1), y(-1))*(dt()/dy()));

            eval(sol(comp(1))) = eval(sol(comp(1)) -
                                      ((ux(y(-1))^2) / hx(y(-1))        +hx(y(-1))*hx(y(-1))*(g()/2.)  -
                                       ((ux(x(-1),y(-1))^2) / hx(x(-1), y(-1)) +(hx(x(-1),y(-1)) ^2)*(g()/2)))*(dt()/dx())  -
                                      (vy(x(-1))       *uy(x(-1))      /hy(x(-1))         -
                                       vy(x(-1), y(-1))*uy(x(-1),y(-1))/hy(x(-1), y(-1)) + hy(x(-1), y(-1))*(g()/2))*(dt()/dy()));

            eval(sol(comp(2))) = eval(sol(comp(2)) -
                                      (ux(y(-1))      *vx(y(-1))        /hy(y(-1)) -
                                      (ux(x(-1),y(-1))*vx(x(-1), y(-1)))/hx(x(-1), y(-1)) )*(dt()/dx())-
                                      ((vy(x(-1))^2)        /hy(x(-1))        +(hy(x(-1))       ^2)*(g()/2) -
                                       (vy(x(-1), y(-1))^2) /hy(x(-1), y(-1)) +(hy(x(-1), y(-1))^2)*(g()/2)   )*(dt()/dy()));

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

    typedef gridtools::BACKEND::temporary_storage_type<float_type, layout_t >::type tmp_storage_type;

    /** -buffer of storages for the time integration (all the components are derived in time)*/
    typedef extend_width</*tmp_*/storage_type::basic_type, 1> extended_tmp_type;

    /** -buffer of storages for the time integration (all the components are derived in time)*/
    typedef extend_width<storage_type::basic_type, 0> extended_type2;

    /** -buffer of 3 storages for the different components (H, U, V), all three derived in time*/
    typedef extend_dim<extended_tmp_type, extended_tmp_type, extended_tmp_type> tmp_type;

    /** -buffer of 3 storages for the different components (H, U, V), all three derived in time*/
    typedef extend_dim<extended_type2, extended_type2, extended_type2> solution_type;

    typedef arg<0, tmp_type > p_tmp;
    typedef arg<1, solution_type > p_sol;
    typedef boost::mpl::vector<p_tmp, p_sol> arg_type_list;

    /** -construct the storage and instantiate the first data field*/
    solution_type sol(d1, d2, d3);

    /** -populate the tmp storage container with all the necessary storage space (6 temporary fields in total)*/
    sol.push_front_new<1>();
    sol.push_front_new<2>();

    tmp_type tmp(d1,d2,d3);
    /** -populate the storage container with all the necessary storage space (3 non temporary fields in total)*/
    tmp.push_front_new<1>();
    tmp.push_front_new<2>();
    tmp.push_front_new<3>();
    tmp.push_front_new<4>();
    tmp.push_front_new<5>();

    gridtools::domain_type<arg_type_list> domain ( (p_sol() = sol), (p_tmp()=tmp) );

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
                gridtools::make_esf<functor_sw>(p_tmp(), p_sol())), // esf_descriptor
            domain, coords
            );

    uint_t T=10;

    model->ready();
    model->steady();

    for (ushort_t i=0; i<T; ++i){
        model->run();
        sol.advance();
    }

}
}

#endif //CXX11
