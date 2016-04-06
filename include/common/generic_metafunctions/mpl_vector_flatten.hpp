#pragma once

#include <boost/mpl/vector.hpp>
#include <boost/mpl/fold.hpp>
#include <boost/mpl/push_back.hpp>

namespace gridtools {

	template <typename T, typename V>
	struct combine {
		typedef typename boost::mpl::fold<
			T,
			V,
			boost::mpl::push_back<boost::mpl::_1, boost::mpl::_2>
		>::type type;
	};

	template<class T>
	struct flatten {
		typedef typename boost::mpl::fold<
			T,
			boost::mpl::vector<>,
			combine<boost::mpl::_2, boost::mpl::_1>
		>::type type;
	};
}
