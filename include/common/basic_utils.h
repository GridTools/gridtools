#pragma once
#include <gridtools.h>
#include "host_device.h"

namespace gridtools {
    GT_FUNCTION
    int_t modulus(int_t __i, int_t __j) {
        return (((((__i%__j)<0)?(__j+__i%__j):(__i%__j))));
    }

#ifdef CXX11_ENABLED

    template <typename T>
    struct storage_sequence_to_tuple;

    template < int N, typename First, typename ... T>
    struct create_tuple{
        typedef typename create_tuple<N-1, First, First, T...>::type type;
    };

    // template <typename First, typename ... T>
    // struct create_tuple< 1, First, T ...  >{
    //     typedef std::tuple<First, First, T...> type;
    // };

    // template <int N, typename T>
    // struct create_tuple<N, T>{
    //     typedef create_tuple< N-1, T, T> type;
    // };


    template < typename U, typename ... T >
    struct create_tuple< 0, U, T ...  >{
        typedef std::tuple< U, T... > type;
    };


    template <typename ... Tuples>
    struct concatenate_tuple{
        typedef decltype(std::tuple_cat(std::declval<Tuples>() ... )) type;
    };

    // template <template<typename T, typename ... Args> class Sequence, typename First, typename ... Others>
    // struct storage_sequence_to_tuple<Sequence<First, Others...> >{
    //     typedef typename concatenate_tuple<typename create_tuple<First::field_dimensions-1, First>::type, typename storage_sequence_to_tuple<Sequence<Others...> >::type>::type type;
    // };


    // template < enumtype::backend Backend,
    //            typename ValueType,
    //            typename Layout,
    //            bool IsTemporary,
    //            short_t FieldDimension
    //            >
    // struct base_storage;

    // template<typename Storage>
    // struct storage_sequence_to_tuple<boost::mpl::v_item<Storage *, boost::mpl::v_item<gridtools::base_storage<enumtype::Host, float,
    //   gridtools::layout_map<0, 1, 2>, true, 1> *, boost::mpl::vector<mpl_::na, mpl_::na, mpl_::na,
    //                                                                  mpl_::na, mpl_::na, mpl_::na, mpl_::na, mpl_::na, mpl_::na, mpl_::na, mpl_::na, mpl_::na,
    //                                                                  mpl_::na, mpl_::na, mpl_::na, mpl_::na, mpl_::na, mpl_::na, mpl_::na, mpl_::na>, 0>, 0> >
    // {
    //     typedef int type;
    // };

    template<typename Storage, typename ... Args>
    struct storage_sequence_to_tuple<boost::mpl::v_item<Storage*, boost::mpl::vector<Args ... >, 0 > >
    {
        typedef typename create_tuple<Storage::field_dimensions-1, typename Storage::value_type*>::type type;
    };

    template<typename Storage, typename Whatever>
    struct storage_sequence_to_tuple<boost::mpl::v_item<Storage *, Whatever, 0> >
    {
        typedef typename concatenate_tuple<typename create_tuple<Storage::field_dimensions-1, typename Storage::value_type*>::type, typename storage_sequence_to_tuple< Whatever >::type>::type type;
    };

#endif
} // namespace gridtools
