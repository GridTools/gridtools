#pragma once

#include "../common/gpu_clone.hpp"
#include "./meta_storage_base.hpp"
#include "./meta_storage_tmp.hpp"
#include "./meta_storage_aligned.hpp"
#ifdef CXX11_ENABLED
#include "../common/generic_metafunctions/repeat_template.hpp"
#include "../common/generic_metafunctions/variadic_to_vector.hpp"
#include <boost/type_traits/is_integral.hpp>
#include <boost/utility/enable_if.hpp>
#include <boost/mpl/vector.hpp>
#include <boost/mpl/at.hpp>
#endif
/**
   @file
   @brief implementation of a container for the storage meta information
*/

/**
   @class
   @brief containing the meta information which is clonable to GPU

   double inheritance is used in order to make the storage metadata clonable to the gpu.

   NOTE: Since this class is inheriting from clonable_to_gpu, the constexpr property of the
   meta_storage_base is lost (this can be easily avoided on the host)
*/
namespace gridtools {

    template < typename BaseStorage >
    struct meta_storage : public BaseStorage, clonable_to_gpu< meta_storage< BaseStorage > > {

        static const bool is_temporary = BaseStorage::is_temporary;
        typedef BaseStorage super;
        typedef typename BaseStorage::basic_type basic_type;
        typedef typename BaseStorage::index_type index_type;
        typedef meta_storage< BaseStorage > original_storage;
        typedef clonable_to_gpu< meta_storage< BaseStorage > > gpu_clone;

        using super::space_dimensions;

        /** @brief copy ctor

            forwarding to the base class
        */
        __device__ meta_storage(meta_storage< BaseStorage > const &other) : super(other) {}

#if defined(CXX11_ENABLED)
        /** @brief ctor

            forwarding to the base class
        */
        template < typename... IntTypes,
            typename Dummy = typename boost::enable_if_c<
                boost::is_integral< typename boost::mpl::at_c< typename variadic_to_vector< IntTypes... >::type,
                    0 >::type >::type::value,
                bool >::type >
        meta_storage(IntTypes... args)
            : super(args...) {}

        constexpr meta_storage(array< uint_t, space_dimensions > const &a) : super(a) {}
#else
        // constructor picked in absence of CXX11 or with GCC<4.9
        /** @brief ctor

            forwarding to the base class
        */
        explicit meta_storage(uint_t dim1, uint_t dim2, uint_t dim3) : super(dim1, dim2, dim3) {}

        /** @brief ctor

            forwarding to the base class
        */
        meta_storage(uint_t const &initial_offset_i,
            uint_t const &initial_offset_j,
            uint_t const &dim3,
            uint_t const &n_i_threads,
            uint_t const &n_j_threads)
            : super(initial_offset_i, initial_offset_j, dim3, n_i_threads, n_j_threads) {}
#endif

#ifndef __CUDACC__
      private:
#endif
        /** @brief empty ctor

            should never be called
            (only by nvcc because it does not compile the parallel_storage CXX11 version)
        */
        explicit meta_storage() : super() {}
    };

    /** \addtogroup specializations Specializations
        Partial specializations
        @{
    */

    template < typename T >
    struct is_meta_storage;

    template < typename Storage >
    struct is_meta_storage< meta_storage< Storage > > : boost::mpl::true_ {};

    template < typename Storage >
    struct is_meta_storage< meta_storage< Storage > & > : boost::mpl::true_ {};

    template < typename Storage >
    struct is_meta_storage< no_meta_storage_type_yet< Storage > > : is_meta_storage< Storage > {};

#ifdef CXX11_ENABLED
    template < ushort_t Index, typename Layout, bool IsTemporary, typename... Whatever >
    struct is_meta_storage< meta_storage_base< Index, Layout, IsTemporary, Whatever... > > : boost::mpl::true_ {};
#else
    template < ushort_t Index, typename Layout, bool IsTemporary, typename TileI, typename TileJ >
    struct is_meta_storage< meta_storage_base< Index, Layout, IsTemporary, TileI, TileJ > > : boost::mpl::true_ {};
#endif

    template < typename T >
    struct is_ptr_to_meta_storage : boost::mpl::false_ {};

    template < typename T >
    struct is_ptr_to_meta_storage< pointer< const T > > : is_meta_storage< T > {};

    /**@}*/

} // namespace gridtools
