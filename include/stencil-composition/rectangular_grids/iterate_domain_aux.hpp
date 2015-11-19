#pragma once
#include "stencil-composition/rectangular_grids/accessor_metafunctions.hpp"

/**
   @file
   @brief file implementing helper functions which are used in iterate_domain to assign/increment strides, access indices and storage pointers.

   All the helper functions use template recursion to implement loop unrolling
*/

namespace gridtools{
/*
    template<typename Accessor, typename CachesMap>
    struct accessor_is_cached
    {
        typedef typename boost::mpl::eval_if<
            is_accessor<Accessor>,
            accessor_index<Accessor>,
            boost::mpl::identity<static_int<-1> >
        >::type accessor_index_t;

        typedef typename boost::mpl::eval_if<
            is_accessor<Accessor>,
            boost::mpl::has_key<
                CachesMap,
                //TODO: ERROR in Clang:
                //non-type template argument evaluates to -1, which cannot be narrowed to type 'uint_t'
#ifdef __CUDACC__
                static_uint<accessor_index_t::value>
#else // the following is NOT correct!! but compiles
                static_int<accessor_index_t::value>
#endif
                >,
            boost::mpl::identity<boost::mpl::false_>
        >::type type;
        BOOST_STATIC_CONSTANT(bool, value=(type::value));
    };
*/

}//namespace gridtools
