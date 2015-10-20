#pragma once
#include "align.hpp"
#include "padding.hpp"
#include "../common/generic_metafunctions/all_integrals.hpp"

namespace gridtools {

    template<typename T>
    struct is_padding;

    template<typename MetaStorageBase
             , typename AlignmentBoundary
             , typename Padding
             >
    struct meta_storage_aligned;


    template<typename MetaStorageBase
             , typename AlignmentBoundary
             , template<ushort_t ... > class Padding
#ifdef CXX11_ENABLED
             , ushort_t ... Pad>
    struct meta_storage_aligned<MetaStorageBase, AlignmentBoundary, Padding<Pad ...> >
#else
        , ushort_t Pad1, ushort_t Pad2, ushort_t Pad3>
        struct meta_storage_aligned<MetaStorageBase, AlignmentBoundary, Padding<Pad1, Pad2, Pad3> >
#endif

        : public MetaStorageBase{

#ifdef CXX11_ENABLED
            typedef Padding<Pad ...> padding_t;//ranges
#else
            typedef Padding<Pad1, Pad2, Pad3> padding_t;//ranges
#endif
            static const ushort_t s_alignment_boundary = AlignmentBoundary::value;

            typedef AlignmentBoundary alignment_boundary_t;
            typedef MetaStorageBase super;



            GRIDTOOLS_STATIC_ASSERT(is_padding<padding_t>::type::value, "wrong type");
            // GRIDTOOLS_STATIC_ASSERT(boost::mpl::size<Padding>::type::value == space_dimensions, "error in the paddindg size");

            /**
               @brief constructor given the space dimensions

               NOTE: this contructor is constexpr, i.e. the storage metadata information could be used
               at compile-time (e.g. in template metafunctions)
            */
#ifdef CXX11_ENABLED
            template <class ... IntTypes
                      , typename Dummy = all_integers<IntTypes...>
                      >
            constexpr meta_storage_aligned(  IntTypes const& ... dims_  ) :
                super(align<s_alignment_boundary>(dims_)+Pad ...)
            {
            }

#else

            // non variadic non constexpr constructor
            meta_storage_aligned(  uint_t const& d1, uint_t const& d2, uint_t const& d3 ) :
                super(align<s_alignment_boundary>(d1)+Pad1, align<s_alignment_boundary>(d2)+Pad2, align<s_alignment_boundary>(d3)+Pad3)
            {
            }
#endif

            /**
               @brief initializing a given coordinate (i.e. multiplying times its stride)

               \param steps_ the input coordinate value
               \param index_ the output index
               \param strides_ the strides array
            */
            template <uint_t Coordinate, typename StridesVector >
            GT_FUNCTION
            static void initialize(uint_t const& steps_, uint_t const& block_, int_t* RESTRICT index_, StridesVector const& RESTRICT strides_){
                uint_t steps_padded_ = steps_+padding_t::template get<Coordinate>();
                super::template initialize<Coordinate>(steps_padded_, block_, index_, strides_ );
            }

        };

    template <typename T>
    struct is_meta_storage;

    template< typename MetaStorageBase, typename Alignment, typename Padding>
    struct is_meta_storage<meta_storage_aligned<MetaStorageBase, Alignment, Padding> > : boost::mpl::true_{};


} // namespace gridtools
