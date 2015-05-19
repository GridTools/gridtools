/*
 * test_local_domain.cpp
 *
 *  Created on: Apr 9, 2015
 *      Author: carlosos
 */

//#define BOOST_NO_CXX11_RVALUE_REFERENCES

#include <gridtools.h>
#include <boost/mpl/equal.hpp>
#include <boost/fusion/include/make_vector.hpp>

#include "gtest/gtest.h"

#include <stencil-composition/backend.h>

using namespace gridtools;
using gridtools::level;
using gridtools::arg_type;
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
        typedef const arg_type<0> in;
        typedef arg_type<1> out;
        typedef boost::mpl::vector<in,out> arg_list;


        template <typename Evaluation>
        GT_FUNCTION
        static void Do(Evaluation const & eval, x_interval) {}
    };

    std::ostream& operator<<(std::ostream& s, dummy_functor const) {
        return s << "dummy_function";
    }

}
template<typename T> struct printo{BOOST_MPL_ASSERT_MSG((false), OOOOOOOOOOOO, (T));};

TEST(test_local_domain, merge_mss_local_domains) {
    using namespace local_domain_stencil;

    typedef layout_map<2,1,0> layout_ijk_t;
    typedef layout_map<0,1,2> layout_kji_t;
    typedef gridtools::backend<Host, Naive >::storage_type<float_type, layout_ijk_t >::type storage_type;
    typedef gridtools::backend<Host, Naive >::storage_type<float_type, layout_kji_t >::type storage_buff_type;

    typedef arg<0, storage_type> p_in;
    typedef arg<1, storage_buff_type> p_buff;
    typedef arg<2, storage_type> p_out;
    typedef boost::mpl::vector<p_in, p_buff, p_out> arg_type_list;

    uint_t d1 = 1;
    uint_t d2 = 1;
    uint_t d3 = 1;

    storage_type in(d1,d2,d3,-3.5,"in");
    storage_buff_type buff(d1,d2,d3,1.5,"buff");
    storage_type out(d1,d2,d3,1.5,"out");

    gridtools::domain_type<arg_type_list> domain((p_in() = in),  (p_buff() = buff), (p_out() = out) );

    uint_t di[5] = {0, 0, 0, d1-1, d1};
    uint_t dj[5] = {0, 0, 0, d2-1, d2};

    gridtools::coordinates<local_domain_stencil::axis> coords(di, dj);
    coords.value_list[0] = 0;
    coords.value_list[1] = d3-1;

    typedef typename decltype(make_computation<gridtools::backend<Host, Naive >, layout_ijk_t>
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

    typedef typename intermediate_backend<intermediate_t>::type backend_t;
    typedef typename intermediate_domain_type<intermediate_t>::type domain_t;
    typedef typename intermediate_mss_array<intermediate_t>::type mss_array_t;
    typedef typename intermediate_layout_type<intermediate_t>::type layout_t;
    typedef typename intermediate_is_stateful<intermediate_t>::type is_stateful_t;

    typedef typename create_actual_arg_list<
        backend_t,
        domain_t,
        mss_array_t,
        float_type,
        layout_t
    >::type actual_arg_list_t;

    typedef typename create_mss_local_domains<
        mss_array_t, domain_t, actual_arg_list_t, is_stateful_t::value
    >::type mss_local_domains_t;

    BOOST_STATIC_ASSERT((boost::mpl::size<mss_local_domains_t>::value==1));

    typedef boost::mpl::front<mss_local_domains_t>::type mss_local_domain_t;
    typedef merge_local_domain_sequence<mss_local_domain_t::LocalDomainList>::type merged_local_domain_t;

    BOOST_STATIC_ASSERT((boost::mpl::size<merged_local_domain_t>::value==1));

    // the merged local domain should contain the args used by all the esfs
    BOOST_STATIC_ASSERT((
        boost::mpl::equal<
            local_domain_esf_args<
                boost::mpl::front<merged_local_domain_t>::type
            >::type,
            boost::mpl::vector3<p_in, p_buff, p_out>
        >::value
    ));

    //the list of storage pointers stored in all local domains are the same. So the merged local domain
    // should also contain the same list of pointer. We check the results against the list of storage pointers
    // of first esf of first msf
    BOOST_STATIC_ASSERT((
        boost::mpl::equal<
            local_domain_storage_pointers<
                boost::mpl::front<merged_local_domain_t>::type
            >::type,
            local_domain_storage_pointers<
                boost::mpl::front<mss_local_domain_t::LocalDomainList>::type
            >::type
        >::value
    ));

    EXPECT_TRUE(true);

    typedef create_args_lookup_map<mss_local_domain_t, merged_local_domain_t>::type mss_esf_args_lookup_map;
    BOOST_STATIC_ASSERT((boost::mpl::size<mss_esf_args_lookup_map>::value == 2));

    BOOST_STATIC_ASSERT((
        boost::mpl::equal<
            boost::mpl::at<mss_esf_args_lookup_map, boost::mpl::int_<0> >::type,
            boost::mpl::map2<
                boost::mpl::pair<boost::mpl::integral_c<int, 0>, boost::mpl::long_<0> >,
                boost::mpl::pair<boost::mpl::integral_c<int, 1>, boost::mpl::long_<1> >
            >
        >::value
    ));
    BOOST_STATIC_ASSERT((
        boost::mpl::equal<
            boost::mpl::at<mss_esf_args_lookup_map, boost::mpl::int_<1> >::type,
            boost::mpl::map2<
                boost::mpl::pair<boost::mpl::integral_c<int, 0>, boost::mpl::long_<1> >,
                boost::mpl::pair<boost::mpl::integral_c<int, 1>, boost::mpl::long_<2> >
            >
        >::value
    ));
}



