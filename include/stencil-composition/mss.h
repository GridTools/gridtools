#pragma once
#include "execution_types.h"
/**
@file
@brief descriptor of the Multi Stage Stencil (MSS)
*/
namespace gridtools {

    /** @brief Descriptors for  Multi Stage Stencil (MSS) */
    template <typename ExecutionEngine,
              typename ArrayEsfDescr>
    struct mss_descriptor {
        template <typename State, typename SubArray>
        struct keep_scanning
          : boost::mpl::fold<
                typename SubArray::esf_list,
                State,
                boost::mpl::push_back<boost::mpl::_1, boost::mpl::_2>
            >
        {};

        template <typename Array>
        struct linearize_esf_array
	    : boost::mpl::fold<
	    Array,
	    boost::mpl::vector<>,
	    boost::mpl::if_<
		is_independent<boost::mpl::_2>,
		keep_scanning<boost::mpl::_1, boost::mpl::_2>,
		boost::mpl::push_back<boost::mpl::_1, boost::mpl::_2>
                >
            >
        {};

        typedef ArrayEsfDescr esf_array; // may contain independent constructs
        typedef ExecutionEngine execution_engine_t;

        /** Collect all esf nodes in the the multi-stage descriptor. Recurse into independent
            esf structs. Independent functors are listed one after the other.*/
        typedef typename linearize_esf_array<esf_array>::type linear_esf;

        /** Compute a vector of vectors of temp indices of temporaries initialized by each functor*/
        typedef typename boost::mpl::fold<linear_esf,
					  boost::mpl::vector<>,
					  boost::mpl::push_back<boost::mpl::_1, get_temps_per_functor<boost::mpl::_2> >
					  >::type written_temps_per_functor;
    };

} // namespace gridtools
