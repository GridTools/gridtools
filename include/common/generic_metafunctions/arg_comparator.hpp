#pragma once
#include "../../gridtools.hpp"
namespace gridtools {

    /**
       @brief struct containing conditionals for several types.

       To be used with e.g. mpl::sort
    */
    struct arg_comparator {
        template < typename T1, typename T2 >
        struct apply;

        /**specialization for storage pairs*/
        template < typename T1, typename T2, typename T3, typename T4 >
        struct apply< arg_storage_pair< T1, T2 >, arg_storage_pair< T3, T4 > >
            : public boost::mpl::bool_< (T1::index_type::value < T3::index_type::value) > {};

        /**specialization for storage placeholders*/
        template < ushort_t I1, typename T1, ushort_t I2, typename T2 >
        struct apply< arg< I1, T1 >, arg< I2, T2 > > : public boost::mpl::bool_< (I1 < I2) > {};

        /**specialization for static integers*/
        template < typename T, T T1, T T2 >
        struct apply< boost::mpl::integral_c< T, T1 >, boost::mpl::integral_c< T, T2 > >
            : public boost::mpl::bool_< (T1 < T2) > {};
    };
}
