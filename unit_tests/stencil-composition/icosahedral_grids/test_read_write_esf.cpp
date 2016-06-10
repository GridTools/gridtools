#include "gtest/gtest.h"
#include <stencil_composition/stencil_composition.hpp>

using namespace gridtools;
using namespace enumtype;

namespace rw_test {

#ifdef __CUDACC__
    using backend_t = ::gridtools::backend< Cuda, GRIDBACKEND, Block >;
#else
#ifdef BACKEND_BLOCK
    using backend_t = ::gridtools::backend< Host, GRIDBACKEND, Block >;
#else
    using backend_t = ::gridtools::backend< Host, GRIDBACKEND, Naive >;
#endif
#endif

    using icosahedral_topology_t = ::gridtools::icosahedral_topology< backend_t >;

    typedef gridtools::interval< level< 0, -1 >, level< 1, -1 > > x_interval;
    typedef gridtools::interval< level< 0, -2 >, level< 1, 1 > > axis;

    struct test_functor {
        typedef accessor< 0, in, icosahedral_topology_t::cells, extent< 0 > > i0;
        typedef accessor< 1, inout, icosahedral_topology_t::cells > o0;
        typedef accessor< 2, in, icosahedral_topology_t::cells, extent< 2 > > i1;
        typedef accessor< 3, inout, icosahedral_topology_t::cells > o1;
        typedef accessor< 4, in, icosahedral_topology_t::cells, extent< 3 > > i2;
        typedef accessor< 5, inout, icosahedral_topology_t::cells > o2;
        typedef accessor< 6, in, icosahedral_topology_t::cells, extent< 4 > > i3;
        typedef accessor< 7, inout, icosahedral_topology_t::cells > o3;

        typedef boost::mpl::vector8< i0, o0, i1, o1, i2, o2, i3, o3 > arg_list;

        template < typename Evaluation >
        GT_FUNCTION static void Do(Evaluation const &eval, x_interval) {}
    };
} // namespace rw_test

using namespace rw_test;

TEST(esf, read_write) {

    typedef gridtools::layout_map< 2, 1, 0 > layout_t;

    using cell_storage_type = typename backend_t::storage_t< icosahedral_topology_t::cells, double >;

    typedef arg< 0, cell_storage_type > pi0;
    typedef arg< 1, cell_storage_type > po0;
    typedef arg< 2, cell_storage_type > pi1;
    typedef arg< 3, cell_storage_type > po1;
    typedef arg< 4, cell_storage_type > pi2;
    typedef arg< 5, cell_storage_type > po2;
    typedef arg< 6, cell_storage_type > pi3;
    typedef arg< 7, cell_storage_type > po3;

    typedef boost::mpl::vector8< pi0, po0, pi1, po1, pi2, po2, pi3, po3 > args;

    typedef esf_descriptor< test_functor, icosahedral_topology_t, icosahedral_topology_t::cells, args > esf_t;

    GRIDTOOLS_STATIC_ASSERT(is_accessor_readonly< test_functor::i0 >::type::value, "");
    GRIDTOOLS_STATIC_ASSERT(!is_accessor_readonly< test_functor::o0 >::type::value, "");

    GRIDTOOLS_STATIC_ASSERT(is_accessor_readonly< test_functor::i1 >::type::value, "");
    GRIDTOOLS_STATIC_ASSERT(!is_accessor_readonly< test_functor::o1 >::type::value, "");

    GRIDTOOLS_STATIC_ASSERT(is_accessor_readonly< test_functor::i2 >::type::value, "");
    GRIDTOOLS_STATIC_ASSERT(!is_accessor_readonly< test_functor::o2 >::type::value, "");

    GRIDTOOLS_STATIC_ASSERT(is_accessor_readonly< test_functor::i3 >::type::value, "");
    GRIDTOOLS_STATIC_ASSERT(!is_accessor_readonly< test_functor::o3 >::type::value, "");

    GRIDTOOLS_STATIC_ASSERT(!is_written< esf_t >::apply< static_int< 0 > >::type::value, "");
    GRIDTOOLS_STATIC_ASSERT(is_written< esf_t >::apply< static_int< 1 > >::type::value, "");

    GRIDTOOLS_STATIC_ASSERT(!is_written< esf_t >::apply< static_int< 2 > >::type::value, "");
    GRIDTOOLS_STATIC_ASSERT(is_written< esf_t >::apply< static_int< 3 > >::type::value, "");

    GRIDTOOLS_STATIC_ASSERT(!is_written< esf_t >::apply< static_int< 4 > >::type::value, "");
    GRIDTOOLS_STATIC_ASSERT(is_written< esf_t >::apply< static_int< 5 > >::type::value, "");

    GRIDTOOLS_STATIC_ASSERT(!is_written< esf_t >::apply< static_int< 6 > >::type::value, "");
    GRIDTOOLS_STATIC_ASSERT(is_written< esf_t >::apply< static_int< 7 > >::type::value, "");

    EXPECT_TRUE(true);
}
