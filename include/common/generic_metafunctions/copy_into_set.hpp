#pragma once
#include <boost/mpl/copy.hpp>
#include <boost/mpl/inserter.hpp>
#include <boost/mpl/set.hpp>

namespace gridtools
{
    //similar to boost::mpl::copy but it copies into an associative set container
    template<typename ToInsert, typename Seq>
    struct copy_into_set
    {
        typedef typename boost::mpl::copy<
            ToInsert,
            boost::mpl::inserter<
                boost::mpl::set0<>, boost::mpl::insert<boost::mpl::_1, boost::mpl::_2>
            >
        >::type type;
    };

} // namespace
