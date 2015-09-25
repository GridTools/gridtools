/*
 * test_local_domain.cpp
 *
 *  Created on: Apr 9, 2015
 *      Author: carlosos
 */

//#define BOOST_NO_CXX11_RVALUE_REFERENCES

#include <gridtools.hpp>
#include <boost/mpl/equal.hpp>
#include <boost/fusion/include/make_vector.hpp>

#include "gtest/gtest.h"

#include "stencil-composition/backend.hpp"
#include "stencil-composition/make_computation.hpp"
#include "stencil-composition/intermediate_metafunctions.hpp"

using namespace gridtools;
using gridtools::level;
using gridtools::accessor;
using gridtools::range;
using gridtools::arg;

using namespace gridtools;
using namespace enumtype;

namespace local_domain_stencil{
    // This is the definition of the special regions in the "vertical" direction
    typedef gridtools::interval<level<0,-1>, level<1,-1> > x_interval;
    typedef gridtools::interval<level<0,-2>, level<1,1> > axis;

    // These are the stencil operators that compose the multistage stencil in this test
    struct dummy_functor {
        typedef const accessor<0> in;
        typedef accessor<1> out;
        typedef boost::mpl::vector<in,out> arg_list;


        template <typename Evaluation>
        GT_FUNCTION
        static void Do(Evaluation const & eval, x_interval) {}
    };

    std::ostream& operator<<(std::ostream& s, dummy_functor const) {
        return s << "dummy_function";
    }

}

TEST(test_local_domain, merge_mss_local_domains) {
    using namespace local_domain_stencil;

    typedef layout_map<2,1,0> layout_ijk_t;
    typedef layout_map<0,1,2> layout_kji_t;
    typedef storage_info< layout_ijk_t> meta_ijk_t;
    typedef storage_info< layout_kji_t> meta_kji_t;
    typedef gridtools::backend<Host, Naive >::storage_type<float_type, meta_ijk_t >::type storage_type;
    typedef gridtools::backend<Host, Naive >::storage_type<float_type, meta_kji_t >::type storage_buff_type;

    typedef arg<0, storage_type> p_in;
    typedef arg<1, storage_buff_type> p_buff;
    typedef arg<2, storage_type> p_out;
    typedef boost::mpl::vector<p_in, p_buff, p_out> accessor_list;

    uint_t d1 = 1;
    uint_t d2 = 1;
    uint_t d3 = 1;

    meta_ijk_t meta_ijk(d1,d2,d3);
    storage_type in(meta_ijk,-3.5,"in");
    meta_kji_t meta_kji(d1,d2,d3);
    storage_buff_type buff(meta_kji,1.5,"buff");
    storage_type out(meta_ijk,1.5,"out");

    gridtools::domain_type<accessor_list> domain((p_in() = in),  (p_buff() = buff), (p_out() = out) );

    uint_t di[5] = {0, 0, 0, d1-1, d1};
    uint_t dj[5] = {0, 0, 0, d2-1, d2};

    gridtools::coordinates<local_domain_stencil::axis> coords(di, dj);
    coords.value_list[0] = 0;
    coords.value_list[1] = d3-1;

    typedef typename decltype(make_computation<gridtools::backend<Host, Naive > >
        (
            gridtools::make_mss // mss_descriptor
            (
                execute<forward>(),
                gridtools::make_esf<local_domain_stencil::dummy_functor>(p_in() ,p_buff()),
                gridtools::make_esf<local_domain_stencil::dummy_functor>(p_buff() ,p_out())
            ),
            domain, coords
        )
    )::element_type intermediate_t;

    typedef intermediate_backend<intermediate_t>::type backend_t;
    typedef intermediate_domain_type<intermediate_t>::type domain_t;
    typedef intermediate_mss_components_array<intermediate_t>::type mss_components_array_t;

    typedef mss_components_array_t::elements mss_elements_t;

    typedef intermediate_mss_local_domains<intermediate_t>::type mss_local_domains_t;

    BOOST_STATIC_ASSERT((boost::mpl::size<mss_local_domains_t>::value==2));

    typedef boost::mpl::front<mss_local_domains_t>::type mss_local_domain1_t;

    BOOST_STATIC_ASSERT((boost::mpl::size<mss_local_domain1_t::unfused_local_domain_sequence_t>::value==1));
    BOOST_STATIC_ASSERT((boost::mpl::size<mss_local_domain1_t::fused_local_domain_sequence_t>::value==1));

    // the merged local domain should contain the args used by all the esfs
    BOOST_STATIC_ASSERT((
        boost::mpl::equal<
            local_domain_esf_args<
                boost::mpl::front<mss_local_domain1_t::unfused_local_domain_sequence_t>::type
            >::type,
            boost::mpl::vector2<p_in, p_buff>
        >::value
    ));

    typedef boost::mpl::at<mss_local_domains_t, boost::mpl::int_<1> >::type mss_local_domain2_t;

    BOOST_STATIC_ASSERT((boost::mpl::size<mss_local_domain2_t::unfused_local_domain_sequence_t>::value==1));
    BOOST_STATIC_ASSERT((boost::mpl::size<mss_local_domain2_t::fused_local_domain_sequence_t>::value==1));

    // the merged local domain should contain the args used by all the esfs
    BOOST_STATIC_ASSERT((
        boost::mpl::equal<
            local_domain_esf_args<
                boost::mpl::front<mss_local_domain2_t::unfused_local_domain_sequence_t>::type
            >::type,
            boost::mpl::vector2<p_buff, p_out>
        >::value
    ));

    EXPECT_TRUE(true);
}
