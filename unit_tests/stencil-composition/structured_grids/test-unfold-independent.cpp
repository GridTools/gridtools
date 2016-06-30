#include "gtest/gtest.h"
#include <stencil-composition/stencil-composition.hpp>

struct functor {
    using a0 = gridtools::accessor< 0, gridtools::enumtype::inout >;
    using a1 = gridtools::accessor< 1, gridtools::enumtype::inout >;

    typedef boost::mpl::vector< a0, a1 > arg_list;
};

struct fake_storage_type {
    using value_type = int;
    using iterator_type = int *;
};

TEST(unfold_independent, test) {

    using namespace gridtools;

    //    typedef gridtools::STORAGE<double, gridtools::layout_map<0,1,2> > storage_type;

    typedef arg< 0, fake_storage_type > p0;
    typedef arg< 1, fake_storage_type > p1;

    using esf_type = decltype(make_stage< functor >(p0(), p1()));

    using mss_type =
        decltype(make_multistage(enumtype::execute< enumtype::forward >(),
            make_stage< functor >(p0(), p1()),
            make_stage< functor >(p0(), p1()),
            make_stage< functor >(p0(), p1()),
            make_independent(make_stage< functor >(p0(), p1()),
                              make_stage< functor >(p0(), p1()),
                              make_independent(make_stage< functor >(p0(), p1()), make_stage< functor >(p0(), p1())))));

    using sequence = unwrap_independent< mss_type::esf_sequence_t >::type;

    GRIDTOOLS_STATIC_ASSERT((boost::mpl::size< sequence >::type::value == 7), "");

    GRIDTOOLS_STATIC_ASSERT((is_sequence_of< sequence, is_esf_descriptor >::value), "");
}
