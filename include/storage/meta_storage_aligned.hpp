#pragma once
#include "align.hpp"
#include "padding.hpp"
#include "../common/generic_metafunctions/all_integrals.hpp"

namespace gridtools {

    template<typename T>
    struct is_padding;

    /**
       @brief decorator of the meta_storage_base class, adding meta-information about the alignment

       \tparam MetaStorageBase the base class, containing strides and dimensions
       \tparam AlignmentBoundary a type containing a the alignment boundary. This value is set by the librari (it is not explicitly exposed to the user) and it depends on the backend implementation. The values for Host and Cuda platforms are 0 and 32 respectively.
       \tparam Padding extra memory space added at the beginning of a specific dimension. This can be used to align an arbitrary iteration point to the alignment boundary. The padding is exposed to the user, and an automatic check triggers an error if the specified padding and the halo region for the corresponding storage (defined by the ranges in the user function) do not match.

     */
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

#if defined(CXX11_ENABLED)
#ifdef __CUDACC__
            //nvcc has problems with constexpr functions
            typedef Padding<align_struct<AlignmentBoundary::value, Pad>::value-Pad ...> padding_t;//paddings
            typedef Padding<Pad ...> halo_t;//ranges
#else
            typedef Padding<align<AlignmentBoundary::value>(Pad)-Pad ...> padding_t;//paddings
            typedef Padding<Pad ...> halo_t;//ranges
#endif
#else
            typedef Padding<align_struct<AlignmentBoundary::value, Pad1>::value - Pad1
                            , align_struct<AlignmentBoundary::value, Pad2>::value - Pad2
                            , align_struct<AlignmentBoundary::value, Pad3>::value - Pad3
            > padding_t;//paddings
            typedef Padding<Pad1, Pad2, Pad3> halo_t;
#endif

// #ifdef CXX11_ENABLED
//             typedef Padding<align<AlignmentBoundary::value>(Pad)-Pad ...> padding_t;//paddings
//             typedef Padding<Pad ...> halo_t;//ranges
// #else
//             typedef Padding<Pad1, Pad2, Pad3> padding_t;//ranges
// #endif
            static const ushort_t s_alignment_boundary = AlignmentBoundary::value;

            typedef AlignmentBoundary alignment_boundary_t;
            typedef MetaStorageBase super;

            GRIDTOOLS_STATIC_ASSERT(is_meta_storage<MetaStorageBase>::type::value, "wrong type");
            GRIDTOOLS_STATIC_ASSERT(is_aligned<alignment_boundary_t>::type::value, "wrong type");
            GRIDTOOLS_STATIC_ASSERT(is_padding<padding_t>::type::value, "wrong type");
            GRIDTOOLS_STATIC_ASSERT(padding_t::size == super::space_dimensions, "error in the paddindg size");

            /**
               @brief constructor given the space dimensions

               NOTE: this contructor is constexpr, i.e. the storage metadata information could be used
               at compile-time (e.g. in template metafunctions)
            */
#ifdef CXX11_ENABLED
            template <class ... IntTypes
#ifndef __CUDACC__//nvcc does not get it
                       , typename Dummy = all_integers<IntTypes...>
#else
                      , typename Dummy = typename boost::enable_if_c<boost::is_integral< typename boost::mpl::at_c<boost::mpl::vector<IntTypes ...>, 0 >::type >::type::value, bool >::type
#endif
                      >
            constexpr meta_storage_aligned(  IntTypes const& ... dims_  ) :
                super(align<s_alignment_boundary>(dims_+Pad) ...)
            {
            }

#else

            // non variadic non constexpr constructor
            meta_storage_aligned(  uint_t const& d1, uint_t const& d2, uint_t const& d3 ) :
                super(align<s_alignment_boundary>(d1+Pad1), align<s_alignment_boundary>(d2+Pad2), align<s_alignment_boundary>(d3+Pad3))
            {


            }
#endif

            //device copy constructor
            GT_FUNCTION
            constexpr meta_storage_aligned( meta_storage_aligned const& other ) :
                super(other){
            }


            /**
               @brief initializing a given coordinate (i.e. multiplying times its stride)

               \param steps_ the input coordinate value
               \param index_ the output index
               \param strides_ the strides array
            */
            template <uint_t Coordinate, typename StridesVector >
            GT_FUNCTION
            static void initialize(uint_t const& steps_, uint_t const& block_, int_t* RESTRICT index_, StridesVector const& RESTRICT strides_){
                uint_t steps_padded_ = steps_+halo_t::template get<Coordinate>();
                super::template initialize<Coordinate>(steps_padded_, block_, index_, strides_ );
            }

        };

    template <typename T>
    struct is_meta_storage;

    template< typename MetaStorageBase, typename Alignment, typename Padding>
    struct is_meta_storage<meta_storage_aligned<MetaStorageBase, Alignment, Padding> > : boost::mpl::true_{};


} // namespace gridtools
