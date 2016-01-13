#pragma once

#include "../common/gpu_clone.hpp"
#include "meta_storage_base.hpp"
#include "meta_storage_tmp.hpp"
#include "meta_storage_aligned.hpp"
#ifdef CXX11_ENABLED
#include "../common/generic_metafunctions/repeat_template.hpp"
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
namespace gridtools{


    template < typename BaseStorage >
    struct meta_storage : public BaseStorage, clonable_to_gpu<meta_storage<BaseStorage> >{

        static const bool is_temporary=BaseStorage::is_temporary;
        typedef BaseStorage super;
        typedef typename BaseStorage::basic_type basic_type;
        typedef typename BaseStorage::index_type index_type;
        typedef meta_storage<BaseStorage> original_storage;
        typedef clonable_to_gpu<meta_storage<BaseStorage> > gpu_clone;

        /** @brief copy ctor

            forwarding to the base class
        */
        __device__
        meta_storage(BaseStorage const& other)
            :  super(other)
        {}

#if defined(CXX11_ENABLED)
        /** @brief ctor

            forwarding to the base class
        */
        template <class ... UIntTypes>
        explicit meta_storage(  UIntTypes const& ... args ): super(args ...)
        {
        }
#else
        //constructor picked in absence of CXX11 or with GCC<4.9
        /** @brief ctor

            forwarding to the base class
        */
        explicit meta_storage(uint_t const& dim1, uint_t const& dim2, uint_t const& dim3): super(dim1, dim2, dim3) {}

        /** @brief ctor

            forwarding to the base class
        */
        meta_storage( uint_t const& initial_offset_i,
                              uint_t const& initial_offset_j,
                              uint_t const& dim3,
                              uint_t const& n_i_threads,
                              uint_t const& n_j_threads)
            : super(initial_offset_i, initial_offset_j, dim3, n_i_threads, n_j_threads){}
#endif

#ifndef __CUDACC__
    private:
#endif
        /** @brief empty ctor

            should never be called
            (only by nvcc because it does not compile the parallel_storage CXX11 version)
        */
        explicit meta_storage(): super(){}

    };


    /**
       @brief syntactic sugar for the metadata type definition

       \tparam Index an index used to differentiate the types also when there's only runtime
       differences (e.g. only the storage dimensions differ)
       \tparam Layout the map of the layout in memory
       \tparam IsTemporary boolean flag set to true when the storage is a temporary one
       \tmaram ... Tiles variadic argument containing the information abount the tiles
       (for the Block strategy)

       syntax example:
       using metadata_t=storage_info<0,layout_map<0,1,2> >

       NOTE: the information specified here will be used at a later stage
       to define the storage meta information (the meta_storage_base type)
    */
// #ifdef CXX11_ENABLED
//     template < ushort_t Index
//                , typename Layout
//                , typename AlignmentBoundary=aligned<0>
//                , typename Padding=typename repeat_template_c<0, Layout::length, padding>::type
//                >
//     using storage_info = meta_storage<meta_storage_aligned<meta_storage_base<Index, Layout, false>, AlignmentBoundary, Padding > >;


    // template < ushort_t Index
    //            , typename Layout
    //            , typename ... ExtraArgs
    //            >
    // struct storage_info;


    // template < ushort_t Index
    //            , typename Layout
    //            >
    // struct storage_info<Index, Layout> : public meta_storage<meta_storage_base<Index, Layout, false> > {

    //     typedef meta_storage<meta_storage_base<Index, Layout, false> > super;

    //     storage_info(uint_t const& d1, uint_t const& d2, uint_t const& d3) : super(d1,d2,d3){}

    //     GT_FUNCTION
    //     storage_info(storage_info const& t) : super(t){}
    // };


    // template < ushort_t Index
    //            , typename Layout
    //            , typename AlignmentBundary
    //            , typename Padding
    //            >
    // struct storage_info<Index, Layout, AlignmentBoundary, Padding> :
    //     public meta_storage<
    //     meta_storage_aligned<meta_storage_base<Index, Layout, false>
    //                          , AlignmentBoundary
    //                          , Padding> >
    // {
    //     typedef meta_storage<
    //         meta_storage_aligned<meta_storage_base<Index, Layout, false>
    //                              , AlignmentBoundary
    //                              , Padding> >  super;

    //     storage_info(uint_t const& d1, uint_t const& d2, uint_t const& d3) : super(d1,d2,d3){}

    //     GT_FUNCTION
    //     storage_info(storage_info const& t) : super(t){}
    // };

// #else

    // template < ushort_t Index
    //            , typename Layout
    //            >
    // struct storage_info : public meta_storage<meta_storage_base<Index, Layout, false, AlignmentBoundary, Padding> > {

    //     typedef meta_storage<meta_storage_base<Index, Layout, false, AlignmentBoundary, Padding> > super;

    //     storage_info(uint_t const& d1, uint_t const& d2, uint_t const& d3) : super(d1,d2,d3){}

    //     GT_FUNCTION
    //     storage_info(storage_info const& t) : super(t){}
    // };


//     template < ushort_t Index
//                , typename Layout
//                , typename AlignmentBundary = aligned<0>
//                , typename Padding = typename repeat_template_c<0, Layout::length, padding>::type
//                >
//     struct storage_info<Index, Layout, AlignmentBoundary, Padding> :
//         public meta_storage<
//         meta_storage_aligned<meta_storage_base<Index, Layout, false>
//                              , AlignmentBoundary
//                              , Padding> >
//     {
//         typedef meta_storage<
//             meta_storage_aligned<meta_storage_base<Index, Layout, false>
//                                  , AlignmentBoundary
//                                  , Padding> >  super;

//         storage_info(uint_t const& d1, uint_t const& d2, uint_t const& d3) : super(d1,d2,d3){}

//         GT_FUNCTION
//         storage_info(storage_info const& t) : super(t){}
//     };

// #endif

/** \addtogroup specializations Specializations
    Partial specializations
    @{
*/

    template <typename T>
    struct is_meta_storage;

    template< typename Storage>
    struct is_meta_storage<meta_storage<Storage> > : boost::mpl::true_{};

    template< typename Storage>
    struct is_meta_storage<meta_storage<Storage>& > : boost::mpl::true_{};

    template< typename Storage>
    struct is_meta_storage<no_meta_storage_type_yet<Storage> > : is_meta_storage<Storage> {};

#ifdef CXX11_ENABLED
    template<ushort_t Index, typename Layout, bool IsTemporary, typename ... Whatever>
    struct is_meta_storage<meta_storage_base<Index, Layout, IsTemporary, Whatever...> > : boost::mpl::true_{};
#else
    template<ushort_t Index, typename Layout, bool IsTemporary, typename TileI, typename TileJ>
    struct is_meta_storage<meta_storage_base<Index, Layout, IsTemporary, TileI, TileJ> > : boost::mpl::true_{};
#endif

    template<typename T>
    struct is_ptr_to_meta_storage : boost::mpl::false_ {};

    template<typename T>
    struct is_ptr_to_meta_storage<pointer<const T> > : is_meta_storage<T> {};

/**@}*/

} // namespace gridtools
