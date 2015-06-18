#pragma once

#include <boost/type_traits/is_const.hpp>

#include "accessor.h"
#include "domain_type.h"

/**
   @file
   @brief Descriptors for Elementary Stencil Function (ESF)
*/
namespace gridtools {

    /**
     * @brief Descriptors for Elementary Stencil Function (ESF)
     */
    template <typename ESF, typename ArgArray, typename Staggering=staggered<0,0,0,0> >
    struct esf_descriptor {
        typedef ESF esf_function;
        typedef ArgArray args;
        typedef Staggering staggering_t;

        //////////////////////Compile time checks ////////////////////////////////////////////////////////////
        //checking that all the placeholders have a different index
        /**
         * \brief Get a sequence of the same type as original_placeholders, containing the indexes relative to the placehoolders
         * note that the static const indexes are transformed into types using mpl::integral_c
         */
        typedef _impl::compute_index_set<typename esf_function::arg_list> check_holes;
        typedef typename check_holes::raw_index_list raw_index_list;
        typedef typename check_holes::index_set index_set;
        static const ushort_t len=check_holes::len;

        //actual check if the user specified placeholder arguments with the same index
        GRIDTOOLS_STATIC_ASSERT((len == boost::mpl::size<index_set>::type::value ),
                "You specified different placeholders with the same index. Check the indexes of the arg_type definitions.")

            //checking if the index list contains holes (a common error is to define a list of types with indexes which are not contiguous)
            typedef typename boost::mpl::find_if<raw_index_list, boost::mpl::greater<boost::mpl::_1, static_int<len-1> > >::type test;
            //check if the index list contains holes (a common error is to define a list of types with indexes which are not contiguous)
            GRIDTOOLS_STATIC_ASSERT((boost::is_same<typename test::type, boost::mpl::void_ >::value) , "the index list contains holes:\n "
            "The numeration of the placeholders is not contiguous. You have to define each arg_type with a unique identifier ranging "
            " from 1 to N without \"holes\".")
            //////////////////////////////////////////////////////////////////////////////////////////////////////
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

    template<typename ESF, typename ArgArray, typename Staggering>
    struct is_esf_descriptor<esf_descriptor<ESF, ArgArray, Staggering> > : boost::mpl::true_{};

    template <typename T>
    struct is_esf_descriptor<independent_esf<T> > : boost::mpl::true_{};


    // Metafunctions
    template <typename Esf>
    struct is_written_temp {
        template <typename Index>
        struct apply {
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
