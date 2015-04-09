#pragma once

#include <boost/type_traits/is_const.hpp>

#include "stencil-composition/arg_type.h"
/**
   @file
   @brief Descriptors for Elementary Stencil Function (ESF)
*/
namespace gridtools {

    /** @brief Descriptors for Elementary Stencil Function (ESF) */
    template <typename ESF, typename ArgArray>
    struct esf_descriptor {
        typedef ESF esf_function;
        typedef ArgArray args;
    };

    template <typename T, typename V>
    std::ostream& operator<<(std::ostream& s, esf_descriptor<T,V> const7) {
        return s << "esf_desctiptor< " << T() << " , somevector > ";
    }

    template <typename ArgArray>
    struct independent_esf {
        typedef ArgArray esf_list;
    };

    template <typename T>
    struct is_independent
      : boost::false_type
    {};

    template <typename T>
    struct is_independent<independent_esf<T> >
      : boost::true_type
    {};

    template <typename T> struct is_esf_descriptor : boost::mpl::false_{};

    template<typename ESF, typename ArgArray>
    struct is_esf_descriptor<esf_descriptor<ESF, ArgArray> > : boost::mpl::true_{};

    template <typename T>
    struct is_esf_descriptor<independent_esf<T> > : boost::mpl::true_{};


    // Metafunctions
    template <typename Esf>
    struct is_written_temp {
        template <typename Index>
        struct apply {
            // TODO: boolean logic, replace with mpl::and_ and mpl::or_
            typedef typename boost::mpl::if_<
                is_plchldr_to_temp<typename boost::mpl::at<typename Esf::args, Index>::type>,
                typename boost::mpl::if_<
                    boost::is_const<typename boost::mpl::at<typename Esf::esf_function::arg_list, Index>::type>,
                    boost::false_type,
                    boost::true_type
                >::type,
                boost::false_type
            >::type type;
        };
    };

    template <typename Esf>
    struct get_arg_index {
        template <typename Index>
        struct apply {
            typedef typename boost::mpl::at<typename Esf::args, Index>::type type;
        };
    };

    template <typename EsfF>
    struct get_temps_per_functor {
        typedef boost::mpl::range_c<uint_t, 0, boost::mpl::size<typename EsfF::args>::type::value> range;
        typedef typename boost::mpl::fold<
            range,
            boost::mpl::vector<>,
            boost::mpl::if_<
                typename is_written_temp<EsfF>::template apply<boost::mpl::_2>,
                boost::mpl::push_back<
                    boost::mpl::_1,
                    typename get_arg_index<EsfF>::template apply<boost::mpl::_2>
                >,
                boost::mpl::_1
            >
        >::type type;
    };

} // namespace gridtools
