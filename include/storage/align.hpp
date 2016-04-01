#pragma once
namespace gridtools {

    /**
       @brief returns the padding to be added at the end of a specific dimension

       used internally in the library to allocate the storages. By default only the stride 1 dimension is padded.
       The stride 1 dimension is identified at compile-time given the layout-map.
     */
    template < ushort_t Alignment, typename LayoutMap >
    struct align {

        // the stride is one when the value in the layout vector is the highest
        template < uint_t Coordinate >
        struct has_stride_one {
            static const bool value =
                (LayoutMap::template at_< Coordinate >::value == vec_max< typename LayoutMap::layout_vector_t >::value);
            typedef typename boost::mpl::bool_< value >::type type;
        };

        //     NOTE: nvcc does not understand that the functor below can be a constant expression
        /** applies the alignment to run-time values*/
        template < uint_t Coordinate, uint_t Halo, uint_t Padding >
        struct do_align {

            GT_FUNCTION
            static constexpr uint_t apply(uint_t const &dimension) {

                typedef static_uint< Halo + Padding > offset;

                // the stride is one when the value in the layout vector is the highest
                return (Alignment && ((dimension + offset::value) % Alignment) && has_stride_one< Coordinate >::value)
                           ? dimension + offset::value + Alignment - ((dimension + offset::value) % Alignment)
                           : dimension + offset::value;
            }
        };
    };

    /**@brief apply alignment to all coordinates regardless of the layout_map*/
    template < ushort_t Alignment, ushort_t Dimension >
    struct align_all {
        static const uint_t value =
            Alignment
                ? (Alignment && (Dimension % Alignment)) ? (Dimension + Alignment - (Dimension % Alignment)) : Dimension
                : Dimension;
    };

    /** @brief wrapper around the alignment boundary

        This class defines a keyword to be used when defining the storage
     */
    template < ushort_t Boundary >
    struct aligned {
        static const ushort_t value = Boundary;
    };

    template < typename T >
    struct is_aligned : boost::mpl::false_ {};

    template < ushort_t T >
    struct is_aligned< aligned< T > > : boost::mpl::true_ {};

} // namespace gridtools
