#pragma once

#include <boost/mpl/vector.hpp>
#include <boost/mpl/clear.hpp>
#include <boost/mpl/fold.hpp>
#include <boost/mpl/joint_view.hpp>

namespace gridtools {

	template<class T>
	struct flatten {
		typedef typename boost::mpl::fold<
			T,
			typename boost::mpl::clear<T>::type,
			boost::mpl::joint_view<boost::mpl::_1, boost::mpl::_2>
		>::type type;
	};

}
