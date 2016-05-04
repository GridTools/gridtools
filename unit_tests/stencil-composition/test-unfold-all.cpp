#include "gtest/gtest.h"
#include <gridtools.hpp>
#include <stencil-composition/stencil-composition.hpp>

typedef gridtools::interval<gridtools::level<0,-1>, gridtools::level<1,-1> > x_interval;
typedef gridtools::interval<gridtools::level<0,-2>, gridtools::level<1,1> > axis;

template<gridtools::uint_t Id>
struct functor{

    typedef gridtools::accessor<0, gridtools::enumtype::inout> a0;
    typedef gridtools::accessor<1, gridtools::enumtype::in> a1;
    typedef boost::mpl::vector2<a0, a1> arg_list;

    template <typename Evaluation>
    GT_FUNCTION
    static void Do(Evaluation const & eval, x_interval) { }
};


#define BACKEND backend<enumtype::Host,  GRIDBACKEND, enumtype::Naive >

TEST(unfold_all, test) {

    using namespace gridtools;

    //    typedef gridtools::STORAGE<double, gridtools::layout_map<0,1,2> > storage_type;

    conditional<0> cond(false);

    grid<axis> grid({0,0,0,1,2},{0,0,0,1,2});
    grid.value_list[0] = 0;
    grid.value_list[1] = 2;


    typedef gridtools::layout_map<2,1,0> layout_t;//stride 1 on i
    typedef BACKEND::storage_info<0, layout_t> meta_data_t;
    typedef BACKEND::storage_type<float_type, meta_data_t >::type storage_t;
    meta_data_t meta_data_(3,3,3);
    storage_t s0(meta_data_, 0., "s0");
    storage_t s1(meta_data_, 0., "s1");

    typedef arg<0, storage_t > p0;
    typedef arg<1, storage_t > p1;

    typedef boost::mpl::vector2<p0, p1> arg_list;
    domain_type< arg_list > domain( (p0() = s0), (p1() = s1) );

    auto mss1 = make_mss
        (
         enumtype::execute<enumtype::forward>(),
         make_esf<functor<0> >(p0(), p1()),
         make_esf<functor<1> >(p0(), p1()),
         make_esf<functor<2> >(p0(), p1()),
         make_independent
         (
          make_esf<functor<3> >(p0(), p1()),
          make_esf<functor<4> >(p0(), p1()),
          make_independent
          (
           make_esf<functor<5> >(p0(), p1()),
           make_esf<functor<6> >(p0(), p1())
           )
          )
         );

    auto mss2 = make_mss
        (
         enumtype::execute<enumtype::forward>(),
         make_esf<functor<7> >(p0(), p1()),
         make_esf<functor<8> >(p0(), p1()),
         make_esf<functor<9> >(p0(), p1()),
         make_independent
         (
          make_esf<functor<10> >(p0(), p1()),
          make_esf<functor<11> >(p0(), p1()),
          make_independent
          (
           make_esf<functor<12> >(p0(), p1()),
           make_esf<functor<13> >(p0(), p1())
           )
          )
         );


    auto comp = make_computation< BACKEND >
        (domain, grid,
         if_(cond, mss1, mss2)
         );

}
