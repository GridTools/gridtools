#pragma once

#include "../common/gpu_clone.hpp"
#include "meta_storage_tmp.hpp"

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
struct meta_storage_derived : public BaseStorage, clonable_to_gpu<meta_storage_derived<BaseStorage> >{

    static const bool is_temporary=BaseStorage::is_temporary;
    typedef BaseStorage super;
    typedef typename BaseStorage::basic_type basic_type;
    typedef typename BaseStorage::index_type index_type;
    typedef meta_storage_derived<BaseStorage> original_storage;
    typedef clonable_to_gpu<meta_storage_derived<BaseStorage> > gpu_clone;

    /** @brief copy ctor

        forwarding to the base class
    */
    __device__
    meta_storage_derived(BaseStorage const& other)
        :  super(other)
        {}

#if defined(CXX11_ENABLED)
    /** @brief ctor

        forwarding to the base class
    */
    template <class ... UIntTypes>
    explicit meta_storage_derived(  UIntTypes const& ... args ): super(args ...)
        {
        }
#else
    //constructor picked in absence of CXX11 or with GCC<4.9
    /** @brief ctor

        forwarding to the base class
    */
    explicit meta_storage_derived(uint_t const& dim1, uint_t const& dim2, uint_t const& dim3): super(dim1, dim2, dim3) {}

    /** @brief ctor

        forwarding to the base class
    */
    meta_storage_derived( uint_t const& initial_offset_i,
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
    explicit meta_storage_derived(): super(){}

};


#ifdef CXX11_ENABLED
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
    template < ushort_t Index
               , typename Layout
               >
    using storage_info = meta_storage_derived<meta_storage_base<Index, Layout, false > >;
#else

    template < ushort_t Index
               , typename Layout
               >
    struct storage_info : public meta_storage_derived<meta_storage_base<Index, Layout, false> > {

        typedef meta_storage_derived<meta_storage_base<Index, Layout, false> > super;

        storage_info(uint_t const& d1, uint_t const& d2, uint_t const& d3) : super(d1,d2,d3){}

        GT_FUNCTION
        storage_info(storage_info const& t) : super(t){}
    };

#endif

/** \addtogroup specializations Specializations
    Partial specializations
    @{
*/

    template< typename Storage>
    struct is_meta_storage<meta_storage_derived<Storage> > : boost::mpl::true_{};

    template< typename Storage>
    struct is_meta_storage<meta_storage_derived<Storage>& > : boost::mpl::true_{};

    template< typename Storage>
    struct is_meta_storage<no_meta_storage_type_yet<Storage> > : is_meta_storage<Storage> {};

#ifdef CXX11_ENABLED
    template<ushort_t Index, typename Layout, bool IsTemporary, typename ... Whatever>
    struct is_meta_storage<meta_storage_base<Index, Layout, IsTemporary, Whatever...> > : boost::mpl::true_{};
#else
    template<ushort_t Index, typename Layout, bool IsTemporary, typename TileI, typename TileJ>
    struct is_meta_storage<meta_storage_base<Index, Layout, IsTemporary, TileI, TileJ> > : boost::mpl::true_{};

    template < ushort_t Index
               , typename Layout
               >
    struct is_meta_storage<storage_info<Index, Layout> > : boost::mpl::true_{};

#endif

    template<typename T>
    struct is_meta_storage_derived : is_meta_storage<typename boost::remove_pointer<T>::type::super>{};


    template<typename T>
    struct is_ptr_to_meta_storage_derived : boost::mpl::false_ {};

    template<typename T>
    struct is_ptr_to_meta_storage_derived<pointer<const T> > : is_meta_storage_derived<T> {};

/**@}*/

} // namespace gridtools
