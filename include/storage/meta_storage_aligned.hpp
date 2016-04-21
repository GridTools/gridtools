#pragma once
#include "align.hpp"
#include "halo.hpp"
#include "common/pair.hpp"
#include "../common/generic_metafunctions/all_integrals.hpp"

namespace gridtools {

    template < typename T >
    struct is_halo;

    /**
       @brief decorator of the meta_storage_base class, adding meta-information about the alignment

       \tparam MetaStorageBase the base class, containing strides and dimensions
       \tparam Alignment a type containing a the alignment boundary.
       This value is set by the library (it is not explicitly exposed to the user)
       and it depends on the backend implementation. The values for Host and Cuda
       platforms are 0 and 32 respectively.
       \tparam Padding extra memory space added at the beginning of a specific dimension.
       This can be used to align an arbitrary iteration point to the alignment boundary.
       The padding is exposed to the user, and an automatic check triggers an error if the
       specified padding and the halo region for the corresponding storage (defined by the
       ranges in the user function) do not match.

     */
    template < typename MetaStorageBase, typename Alignment, typename HaloType >
    struct meta_storage_aligned;

    template < typename MetaStorageBase,
        typename Alignment
#ifdef CXX11_ENABLED
        ,
        template < ushort_t... P > class HaloType,
        ushort_t... Halo >
    struct meta_storage_aligned< MetaStorageBase, Alignment, HaloType< Halo... > >
#else
        ,
        template < ushort_t, ushort_t, ushort_t > class HaloType,
        ushort_t Halo1,
        ushort_t Halo2,
        ushort_t Halo3 >
    struct meta_storage_aligned< MetaStorageBase, Alignment, HaloType< Halo1, Halo2, Halo3 > >
#endif
        : public MetaStorageBase {

#if defined(CXX11_ENABLED)
        // nvcc has problems with constexpr functions
        typedef HaloType< Halo... > halo_t;                                                 // ranges
        typedef HaloType< align_all< Alignment::value, Halo >::value - Halo... > padding_t; // paddings
#else
        typedef HaloType< align_all< Alignment::value, Halo1 >::value - Halo1,
            align_all< Alignment::value, Halo2 >::value - Halo2,
            align_all< Alignment::value, Halo3 >::value - Halo3 > padding_t; // paddings
        typedef HaloType< Halo1, Halo2, Halo3 > halo_t;
#endif

        static const ushort_t s_alignment = Alignment::value;

        typedef MetaStorageBase super;
        typedef align< s_alignment, typename super::layout > align_t;
        typedef Alignment alignment_t;
        typedef typename MetaStorageBase::basic_type basic_type;
        typedef typename MetaStorageBase::index_type index_type;

        GRIDTOOLS_STATIC_ASSERT(is_meta_storage< MetaStorageBase >::type::value, "wrong type");
        GRIDTOOLS_STATIC_ASSERT(is_aligned< alignment_t >::type::value, "wrong type");
        GRIDTOOLS_STATIC_ASSERT(is_halo< padding_t >::type::value, "wrong type");
        GRIDTOOLS_STATIC_ASSERT(padding_t::size == super::space_dimensions, "error in the paddindg size");

/**
   @brief metafunction for conditional compilation

   returns padding_t if the ID template argument corresponds with the stride 1 dimension,
   halo_t otherwise.
 */
#ifdef CXX11_ENABLED
        template < uint_t ID >
        struct cond : boost::mpl::if_c< align_t::template has_stride_one< ID >::value,
                          static_uint< padding_t::template get< ID >() >,
                          static_uint< 0 > >::type {};
#else
        template < uint_t ID >
        struct cond
            : boost::mpl::if_c< align_t::template has_stride_one< ID >::value, padding_t, halo< 0, 0, 0 > >::type {};
#endif

#ifdef CXX11_ENABLED
        /** metafunction to select the dimension with stride 1 and align it */
        template < uint_t U >
        using lambda_t =
            typename align_t::template do_align< U, (halo_t::template get< U >()), (padding_t::template get< U >()) >;
#endif
/**
   @brief constructor given the space dimensions

   NOTE: this contructor is constexpr, i.e. the storage metadata information could be used
   at compile-time (e.g. in template metafunctions)

   applying 'align' to the integer sequence from 1 to space_dimensions.
   It will select the dimension with stride 1 and align it

   it instanitates a class like
   super(lambda<0>::apply(d1), lambda<1>::apply(d2), lambda<2>::apply(d3), ...)
*/
#ifdef CXX11_ENABLED
        template < class... IntTypes
#ifndef __CUDACC__ // nvcc does not get it
            ,
            typename Dummy = all_integers< IntTypes... >
#else
            ,
            typename Dummy = typename boost::enable_if_c<
                boost::is_integral<
                    typename boost::mpl::at_c< boost::mpl::vector< IntTypes... >, 0 >::type >::type::value,
                bool >::type
#endif
            >
        GT_FUNCTION constexpr meta_storage_aligned(IntTypes... dims_)
            : super(apply_gt_integer_sequence< typename make_gt_integer_sequence< uint_t,
                      sizeof...(IntTypes) >::type >::template apply_zipped< super, lambda_t >(dims_...)) {
        }

        /**@brief Constructor taking an array with the storage dimensions

           forwarding to the constructor below
         */
        template < class Array, typename boost::enable_if< is_array< Array >, int >::type = 0 >
        GT_FUNCTION constexpr meta_storage_aligned(Array const &dims_)
            : meta_storage_aligned(dims_, typename make_gt_integer_sequence< ushort_t, Array::n_dimensions >::type()) {
            GRIDTOOLS_STATIC_ASSERT(is_array< Array >::value, "type");
        }

        /**@brief Constructor taking an array with the storage dimensions

           forwarding to the variadic constructor
         */
        template < typename T, ushort_t... Ids >
        GT_FUNCTION constexpr meta_storage_aligned(
            array< T, sizeof...(Ids) > const &dims_, gt_integer_sequence< ushort_t, Ids... > x_)
            : meta_storage_aligned(dims_[Ids]...) {}

        /**@brief extra level of indirection necessary for zipping the indices*/
        template < typename... UInt, ushort_t... IdSequence >
        GT_FUNCTION constexpr int_t index_(
            gt_integer_sequence< ushort_t, IdSequence... > t, UInt const &... args_) const {

            GRIDTOOLS_STATIC_ASSERT(sizeof...(IdSequence) == sizeof...(UInt),
                "number of arguments used to compute the storage index does not match the storage dimension. Check "
                "that you are accessing the storage correctly.");
            return super::index(args_ + cond< IdSequence >::value...);
        }

        /**@brief just forwarding the index computation to the base class*/
        template < typename T, size_t N >
        GT_FUNCTION int_t index(array< T, N > const &t) const {

            return super::index(t);
        }

        /**@brief */
        template < typename... UInt >
        GT_FUNCTION constexpr int_t index(uint_t const &first_, UInt const &... args_) const {

            /**this call zips 2 variadic packs*/
            return index_(typename make_gt_integer_sequence< ushort_t, sizeof...(Halo) >::type(), first_, args_...);
        }
#else

        /* applying 'align' to the integer sequence from 1 to space_dimensions.
           It will select the dimension with stride 1 and align it*/
        // non variadic non constexpr constructor
        GT_FUNCTION
        meta_storage_aligned(uint_t const &d1, uint_t const &d2, uint_t const &d3)
            : super(align_t::template do_align< 0, (halo_t::s_pad1), (padding_t::s_pad1) >::apply(d1),
                  align_t::template do_align< 1, (halo_t::s_pad2), (padding_t::s_pad2) >::apply(d2),
                  align_t::template do_align< 2, (halo_t::s_pad3), (padding_t::s_pad3) >::apply(d3)) {}

        /**@brief straightforward interface*/
        GT_FUNCTION
        int_t index(uint_t const &i, uint_t const &j, uint_t const &k) const {
            return super::index(i + cond< 0 >::template get< 0 >(),
                j + cond< 1 >::template get< 1 >(),
                k + cond< 2 >::template get< 2 >());
        }

#endif
        // device copy constructor
        GT_FUNCTION
        constexpr meta_storage_aligned(meta_storage_aligned const &other) : super(other) {}

        // empty constructor
        GT_FUNCTION
        constexpr meta_storage_aligned() {}

        /**
           @brief initializing a given coordinate (i.e. multiplying times its stride)

           \param steps_ the input coordinate value
           \param index_ the output index
           \param strides_ the strides array
        */
        template < uint_t Coordinate, typename StridesVector >
        GT_FUNCTION static void initialize(uint_t const &steps_,
            uint_t const &block_,
            int_t *RESTRICT index_,
            StridesVector const &RESTRICT strides_) {
            uint_t steps_padded_ = steps_ +
#ifdef CXX11_ENABLED
                                   cond< Coordinate >::value;
#else
                                   cond< Coordinate >::template get< Coordinate >();
#endif
#ifndef NDEBUG
#ifdef DEBUG_META_STORAGE
#ifdef __CUDACC__
            if (threadIdx.x == 0) {
#endif
                if (align_t::template has_stride_one< Coordinate >::value) {
                    printf("%d, is aligned?\n", steps_padded_);
                    printf("padding + steps: %d\n", steps_padded_);
                }
#ifdef __CUDACC__
            }
#endif
#endif
#endif
            super::template initialize< Coordinate >(steps_padded_, block_, index_, strides_);
        }
    };

    template < typename T >
    struct is_meta_storage;

    template < typename MetaStorageBase, typename Alignment, typename HaloType >
    struct is_meta_storage< meta_storage_aligned< MetaStorageBase, Alignment, HaloType > > : boost::mpl::true_ {};

} // namespace gridtools
