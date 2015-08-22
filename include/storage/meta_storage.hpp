#pragma once
#include "meta_storage_base.hpp"
#include "meta_storage_tmp.hpp"

namespace gridtools{
template < typename BaseStorage >
struct meta_storage_wrapper : public BaseStorage, clonable_to_gpu<meta_storage_wrapper<BaseStorage> >
    {
        static const bool is_temporary=BaseStorage::is_temporary;
        typedef BaseStorage super;
        typedef typename BaseStorage::basic_type basic_type;
        typedef typename BaseStorage::index_type index_type;
        typedef meta_storage_wrapper<BaseStorage> original_storage;
        typedef clonable_to_gpu<meta_storage_wrapper<BaseStorage> > gpu_clone;

        __device__
        meta_storage_wrapper(BaseStorage const& other)
            :  super(other)
            {}

#if defined(CXX11_ENABLED)
        //arbitrary dimensional field
        template <class ... UIntTypes>
        explicit meta_storage_wrapper(  UIntTypes const& ... args ): super(args ...)
            {
            }
#else
        //constructor picked in absence of CXX11 or which GCC<4.9
        explicit meta_storage_wrapper(uint_t const& dim1, uint_t const& dim2, uint_t const& dim3): super(dim1, dim2, dim3) {}

        meta_storage_wrapper( uint_t const& initial_offset_i,
                              uint_t const& initial_offset_j,
                              uint_t const& dim3,
                              uint_t const& n_i_threads,
                              uint_t const& n_j_threads)
            : super(initial_offset_i, initial_offset_j, dim3, n_i_threads, n_j_threads){}
#endif



//    private :
        explicit meta_storage_wrapper(): super(){}
        // BaseStorage m_meta_storage;

    };


#ifdef CXX11_ENABLED
    template < ushort_t Index
               , typename Layout
               , bool IsTemporary
               , typename ... Tiles
               >
    using meta_storage = meta_storage_wrapper<meta_storage_base<Index, Layout, IsTemporary, Tiles...> >;
#else

    template < uint_t Tile, uint_t Minus, uint_t Plus >
    struct tile;

    template < ushort_t Index
               , typename Layout
               , bool IsTemporary
               , typename TileI=int
               , typename TileJ=int
               >
    struct meta_storage;

    template < ushort_t Index
               , typename Layout
               , bool IsTemporary
               , uint_t TileI,uint_t MinusI,uint_t PlusI
               , uint_t TileJ,uint_t MinusJ,uint_t PlusJ
               >
    struct meta_storage<Index, Layout, IsTemporary, tile<TileI,MinusI,PlusI>, tile<TileJ,MinusJ,PlusJ> > : public meta_storage_wrapper<meta_storage_base<Index, Layout, IsTemporary, tile<TileI,MinusI,PlusI>, tile<TileJ,MinusJ,PlusJ> > >{
        typedef meta_storage_wrapper<meta_storage_base<Index, Layout, IsTemporary, tile<TileI,MinusI,PlusI>, tile<TileJ,MinusJ,PlusJ> > > super;

        meta_storage(uint_t const& d1, uint_t const& d2, uint_t const& d3) : super(d1,d2,d3){}

        meta_storage( uint_t const& initial_offset_i,
                      uint_t const& initial_offset_j,
                      uint_t const& dim3,
                      uint_t const& n_i_threads=1,
                      uint_t const& n_j_threads=1)
            : super(initial_offset_i, initial_offset_j, dim3, n_i_threads, n_j_threads){}

        template <typename T>
        GT_FUNCTION
        meta_storage(T const& t) : super(t){}
    };

    template < ushort_t Index
               , typename Layout
               , bool IsTemporary
               >
    struct meta_storage<Index, Layout, IsTemporary, int, int> : public meta_storage_wrapper<meta_storage_base<Index, Layout, IsTemporary> >{
        typedef meta_storage_wrapper<meta_storage_base<Index, Layout, IsTemporary> > super;

        meta_storage(uint_t const& d1, uint_t const& d2, uint_t const& d3) : super(d1,d2,d3){}

        template <typename T>
        GT_FUNCTION
        meta_storage(T const& t) : super(t){}
    };
#endif

    template<typename T>
    struct is_meta_storage : boost::mpl::false_{};

    template< typename Storage>
    struct is_meta_storage<meta_storage_wrapper<Storage> > : boost::mpl::true_{};

    template< typename Storage>
    struct is_meta_storage<no_meta_storage_type_yet<Storage> > : is_meta_storage<Storage> {};

    template < ushort_t Index
               , typename Layout
               , bool IsTemporary
#ifdef CXX11_ENABLED
               , typename ... Tiles
#else
               , typename TileI, typename TileJ
#endif
               >
    struct is_meta_storage<meta_storage
                           <Index, Layout, IsTemporary,
#ifdef CXX11_ENABLED
                            Tiles...
#else
                            TileI, TileJ
#endif
                            > > : boost::mpl::true_{};

#ifdef CXX11_ENABLED
    template<ushort_t Index, typename Layout, bool IsTemporary, typename ... Whatever>
    struct is_meta_storage<meta_storage_base<Index, Layout, IsTemporary, Whatever...> > : boost::mpl::true_{};
#else
    template<ushort_t Index, typename Layout, bool IsTemporary, typename TileI, typename TileJ>
    struct is_meta_storage<meta_storage_base<Index, Layout, IsTemporary, TileI, TileJ> > : boost::mpl::true_{};
#endif

    template<typename T>
    struct is_meta_storage_wrapper : is_meta_storage<typename boost::remove_pointer<T>::type::super>{};


    template<typename T>
    struct is_ptr_to_meta_storage_wrapper : boost::mpl::false_ {};

    template<typename T>
    struct is_ptr_to_meta_storage_wrapper<pointer<const T> > : is_meta_storage_wrapper<T> {};


} // namespace gridtools
