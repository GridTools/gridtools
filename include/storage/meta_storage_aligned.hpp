#pragma once
#include "align.hpp"
#include "halo.hpp"
#include "../common/generic_metafunctions/all_integrals.hpp"

namespace gridtools {

    template<typename T>
    struct is_halo;

    /**
       @brief decorator of the meta_storage_base class, adding meta-information about the alignment

       \tparam MetaStorageBase the base class, containing strides and dimensions
       \tparam AlignmentBoundary a type containing a the alignment boundary. This value is set by the librari (it is not explicitly exposed to the user) and it depends on the backend implementation. The values for Host and Cuda platforms are 0 and 32 respectively.
       \tparam Padding extra memory space added at the beginning of a specific dimension. This can be used to align an arbitrary iteration point to the alignment boundary. The padding is exposed to the user, and an automatic check triggers an error if the specified padding and the halo region for the corresponding storage (defined by the ranges in the user function) do not match.

     */
    template<typename MetaStorageBase
             , typename AlignmentBoundary
             , typename Halo
             >
    struct meta_storage_aligned;


    template<typename MetaStorageBase
             , typename AlignmentBoundary
#ifdef CXX11_ENABLED
             , template<ushort_t ... P> class  Halo
             , ushort_t ... Pad>
    struct meta_storage_aligned<MetaStorageBase, AlignmentBoundary, Halo<Pad ...> >
#else
        , template<ushort_t, ushort_t, ushort_t > class  Halo
        , ushort_t Pad1, ushort_t Pad2, ushort_t Pad3>
        struct meta_storage_aligned<MetaStorageBase, AlignmentBoundary, Halo<Pad1, Pad2, Pad3> >
#endif
        : public MetaStorageBase
        {

#if defined(CXX11_ENABLED)
            //nvcc has problems with constexpr functions
            typedef Halo<Pad ...> halo_t;//ranges
            typedef Halo<align_all<AlignmentBoundary::value, Pad>::value-Pad ...> padding_t;//paddings
#else
            typedef Halo<align_all<AlignmentBoundary::value, Pad1>::value - Pad1
                            , align_all<AlignmentBoundary::value, Pad2>::value - Pad2
                            , align_all<AlignmentBoundary::value, Pad3>::value - Pad3
            > padding_t;//paddings
            typedef Halo<Pad1, Pad2, Pad3> halo_t;
#endif

            static const ushort_t s_alignment_boundary = AlignmentBoundary::value;

            typedef AlignmentBoundary alignment_boundary_t;
            typedef MetaStorageBase super;
            typedef typename MetaStorageBase::basic_type basic_type;
            typedef typename MetaStorageBase::index_type index_type;

            GRIDTOOLS_STATIC_ASSERT(is_meta_storage<MetaStorageBase>::type::value, "wrong type");
            GRIDTOOLS_STATIC_ASSERT(is_aligned<alignment_boundary_t>::type::value, "wrong type");
            GRIDTOOLS_STATIC_ASSERT(is_halo<padding_t>::type::value, "wrong type");
            GRIDTOOLS_STATIC_ASSERT(padding_t::size == super::space_dimensions, "error in the paddindg size");

#ifdef CXX11_ENABLED
            /** metafunction to select the dimension with stride 1 and align it */
            template<uint_t U>
            using lambda_t = typename align<s_alignment_boundary, typename super::layout>::template do_align<U>;
#endif
            /**
               @brief constructor given the space dimensions

               NOTE: this contructor is constexpr, i.e. the storage metadata information could be used
               at compile-time (e.g. in template metafunctions)

               applying 'align' to the integer sequence from 1 to space_dimensions. It will select the dimension with stride 1 and align it

               it instanitates a class like
               super(lambda<0>::apply(d1), lambda<1>::apply(d2), lambda<2>::apply(d3), ...)
            */
#ifdef CXX11_ENABLED
            template <class ... IntTypes
#ifndef __CUDACC__//nvcc does not get it
                      , typename Dummy = all_integers<IntTypes...>
#else
                      , typename Dummy = typename boost::enable_if_c<
                            boost::is_integral<
                                typename boost::mpl::at_c<
                                    boost::mpl::vector<IntTypes ...>, 0 >::type
                                >::type::value, bool
                            >::type
#endif
                      >
            GT_FUNCTION
            constexpr meta_storage_aligned(  IntTypes const& ... dims_  ) :
                super(apply_gt_integer_sequence
                      <typename make_gt_integer_sequence<uint_t, sizeof ... (IntTypes)>::type >::template apply_zipped
                      <super, lambda_t >(dims_ + Pad ...) )
            {
            }

            /**@brief Constructor taking an array with the storage dimensions

               forwarding to the constructor below
             */
            template <class Array, typename boost::enable_if<is_array<Array>, int >::type = 0 >
            GT_FUNCTION
            constexpr meta_storage_aligned( Array const& dims_ ) :
                meta_storage_aligned(dims_, typename make_gt_integer_sequence< ushort_t, 3 >::type())
            {
                GRIDTOOLS_STATIC_ASSERT(is_array<Array>::value, "type");
            }

            /**@brief Constructor taking an array with the storage dimensions

               forwarding to the variadic constructor
             */
            template <typename T, ushort_t ... Ids  >
            GT_FUNCTION
            constexpr
            meta_storage_aligned( array<T, sizeof...(Ids)> const& dims_, gt_integer_sequence<ushort_t, Ids ...> x_ ) :
                meta_storage_aligned(dims_[Ids]...)
            {
            }

            /**@brief extra level of indirection necessary for zipping the indices*/
            template <typename ... UInt, ushort_t ... IdSequence>
            GT_FUNCTION
            uint_t index_( gt_integer_sequence<ushort_t, IdSequence...> t, UInt const& ... args_
                ) const {

                return super::index(args_ + Pad ...);
            }

           /**@brief just forwarding the index computation to the base class*/
            template <typename ... Types>
            GT_FUNCTION
            uint_t index( Types const& ... t) const {

                return super::index(t ...);
            }

            /**@brief */
            template <typename ... UInt, typename ... IdSequence>
            GT_FUNCTION
            uint_t index(uint_t const& first_, UInt const& ... args_) const {

                /**this calls zippes 2 variadic packs*/
                return index_(typename make_gt_integer_sequence<ushort_t, sizeof ... (Pad)>::type(), first_, args_ ... );
                    }
#else

            /* applying 'align' to the integer sequence from 1 to space_dimensions. It will select the dimension with stride 1 and align it*/
            // non variadic non constexpr constructor
            GT_FUNCTION
            meta_storage_aligned(  uint_t const& d1, uint_t const& d2, uint_t const& d3 ) :
                super(align<s_alignment_boundary, typename super::layout>::template do_align<0>::apply(d1+Pad1)
                      , align<s_alignment_boundary, typename super::layout>::template do_align<1>::apply(d2+Pad2)
                      , align<s_alignment_boundary, typename super::layout>::template do_align<2>::apply(d3+Pad3))
            {
            }

            /**@brief straightforward interface*/
            GT_FUNCTION
            uint_t index(uint_t const& i, uint_t const& j, uint_t const&  k) const { return super::index(i+Pad1, j+Pad2, k+Pad3); }


#endif

            //device copy constructor
            GT_FUNCTION
            constexpr meta_storage_aligned( meta_storage_aligned const& other ) :
                super(other){
            }

            //empty constructor
            GT_FUNCTION
            constexpr meta_storage_aligned() {}

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

    template< typename MetaStorageBase, typename Alignment, typename Halo>
    struct is_meta_storage<meta_storage_aligned<MetaStorageBase, Alignment, Halo> > : boost::mpl::true_{};


} // namespace gridtools
