#pragma once

#include <boost/fusion/adapted/mpl.hpp>
#include <boost/fusion/sequence/intrinsic/at.hpp>
#include <boost/fusion/view/zip_view.hpp>
#include <boost/fusion/include/for_each.hpp>
#include <boost/fusion/include/as_set.hpp>

#include <boost/fusion/include/as_vector.hpp>
#include <boost/fusion/include/vector.hpp>
#include <boost/fusion/include/at_c.hpp>
#include <boost/fusion/include/at.hpp>
#include <boost/mpl/at.hpp>
#include <boost/fusion/include/size.hpp>
#include <boost/fusion/adapted/mpl.hpp>
#include <boost/fusion/container/vector.hpp>
#include <boost/fusion/include/vector.hpp>
#include <boost/fusion/container/vector/vector_fwd.hpp>
#include <boost/fusion/include/vector_fwd.hpp>
#include <boost/fusion/container/generation/make_vector.hpp>
#include <boost/fusion/include/make_vector.hpp>
#include <boost/fusion/sequence/io.hpp>
#include <boost/fusion/include/io.hpp>

#include <common/array.hpp>
#include "../../common/gt_assert.hpp"
#include <boost/mpl/vector.hpp>
#include "location_type.hpp"
#include "common/array_addons.hpp"

namespace gridtools {

    // TODO this is duplicated below

    namespace {
        using cells = location_type< 0, 2 >;
        using edges = location_type< 1, 3 >;
        using vertexes = location_type< 2, 1 >;
    }

    template < typename T, typename ValueType >
    struct return_type {
        typedef array< ValueType, 0 > type;
    };

    // static triple dispatch
    template < typename Location1 >
    struct from {
        template < typename Location2 >
        struct to {
            template < typename Color >
            struct with_color;
        };
    };

    template < typename ValueType >
    struct return_type< from< cells >::template to< cells >, ValueType > {
        typedef array< ValueType, 3 > type;
    };

    template < typename ValueType >
    struct return_type< from< cells >::template to< edges >, ValueType > {
        typedef array< ValueType, 3 > type;
    };

    template < typename ValueType >
    struct return_type< from< cells >::template to< vertexes >, ValueType > {
        typedef array< ValueType, 3 > type;
    };

    template < typename ValueType >
    struct return_type< from< edges >::template to< edges >, ValueType > {
        typedef array< ValueType, 4 > type;
    };

    template < typename ValueType >
    struct return_type< from< edges >::template to< cells >, ValueType > {
        typedef array< ValueType, 2 > type;
    };

    template < typename ValueType >
    struct return_type< from< edges >::template to< vertexes >, ValueType > {
        typedef array< ValueType, 2 > type;
    };

    template < typename ValueType >
    struct return_type< from< vertexes >::template to< vertexes >, ValueType > {
        typedef array< ValueType, 6 > type;
    };

    template < typename ValueType >
    struct return_type< from< vertexes >::template to< cells >, ValueType > {
        typedef array< ValueType, 6 > type;
    };

    template < typename ValueType >
    struct return_type< from< vertexes >::template to< edges >, ValueType > {
        typedef array< ValueType, 6 > type;
    };

    template < uint_t SourceColor >
    struct get_connectivity_offset {

        template < int Idx >
        struct get_element {
            GT_FUNCTION
            constexpr get_element() {}

            template < typename Offsets >
            GT_FUNCTION constexpr static array< uint_t, 4 > apply(array< uint_t, 3 > const &i, const Offsets offsets) {
                return {i[0] + offsets[Idx][0],
                    SourceColor + offsets[Idx][1],
                    i[1] + offsets[Idx][2],
                    i[2] + offsets[Idx][3]};
            }
        };
    };

    template < typename T >
    struct is_grid_topology;

    template < typename DestLocation, typename Grid, uint_t SourceColor >
    struct get_connectivity_index {

        GRIDTOOLS_STATIC_ASSERT((is_grid_topology< Grid >::value), "Error");
        GRIDTOOLS_STATIC_ASSERT((is_location_type< DestLocation >::value), "Error");

        template < int Idx >
        struct get_element {
            GT_FUNCTION
            constexpr get_element() {}

            template < typename Offsets >
            GT_FUNCTION static uint_t apply(Grid const &grid, array< uint_t, 3 > const &i, const Offsets offsets) {
                return boost::fusion::at_c< DestLocation::value >(grid.virtual_storages())
                    .index(get_connectivity_offset< SourceColor >::template get_element< Idx >::apply(i, offsets));
            }
        };
    };

    template <>
    template <>
    template <>
    struct from< cells >::to< cells >::with_color< static_int< 1 > > {

        template < typename ValueType >
        using return_t = typename return_type< from< cells >::to< cells >, ValueType >::type;

        /*
         * neighbors order
         *
         *      ____________
         *      \    /\    /
         *       \1 /  \2 /
         *        \/____\/
         *         \  0 /
         *          \  /
         *           \/
         */
        GT_FUNCTION
        constexpr static return_t< array< int_t, 4 > > offsets() {
            return return_t< array< int_t, 4 > >{{{1, -1, 0, 0}, {0, -1, 0, 0}, {0, -1, 1, 0}}};
        }
    };

    template <>
    template <>
    template <>
    struct from< cells >::to< cells >::with_color< static_int< 0 > > {

        template < typename ValueType >
        using return_t = typename return_type< from< cells >::to< cells >, ValueType >::type;

        /*
         * neighbors order
         *
         *        /\
         *       /0 \
         *      /____\
         *     /\    /\
         *    /2 \  /1 \
         *   /____\/____\
         */
        GT_FUNCTION
        constexpr static return_t< array< int_t, 4 > > offsets() {
            return return_t< array< int_t, 4 > >{{{-1, 1, 0, 0}, {0, 1, 0, 0}, {0, 1, -1, 0}}};
        }
    };

    template <>
    template <>
    template <>
    struct from< vertexes >::to< vertexes >::with_color< static_int< 0 > > {

        template < typename ValueType >
        using return_t = typename return_type< from< vertexes >::to< vertexes >, ValueType >::type;

        /*
         * neighbors order
         *
         *          1____2
         *         /\    /\
         *        /  \  /  \
         *       0____\/____3
         *       \    /\    /
         *        \  /  \  /
         *         \5____4/
         *
         */
        GT_FUNCTION
        constexpr static return_t< array< int_t, 4 > > offsets() {
            return return_t< array< int_t, 4 > >{
                {{0, 0, -1, 0}, {-1, 0, 0, 0}, {-1, 0, 1, 0}, {0, 0, 1, 0}, {1, 0, 0, 0}, {1, 0, -1, 0}}};
        }
    };

    template <>
    template <>
    template <>
    struct from< edges >::to< edges >::with_color< static_int< 0 > > {

        template < typename ValueType >
        using return_t = typename return_type< from< edges >::to< edges >, ValueType >::type;

        /*
         * neighbors order
         *
         *        __1___
         *       /\    /
         *      0  \  2
         *     /_3__\/
         */
        GT_FUNCTION
        constexpr static return_t< array< int_t, 4 > > offsets() {
            return return_t< array< int_t, 4 > >{{{0, 2, -1, 0}, {0, 1, 0, 0}, {0, 2, 0, 0}, {1, 1, -1, 0}}};
        }
    };

    template <>
    template <>
    template <>
    struct from< edges >::to< edges >::with_color< static_int< 1 > > {

        template < typename ValueType >
        using return_t = typename return_type< from< edges >::to< edges >, ValueType >::type;

        /*
         * neighbors order
         *
         *      /\
         *     0  1
         *    /____\
         *    \    /
         *     3  2
         *      \/
         */
        GT_FUNCTION
        constexpr static return_t< array< int_t, 4 > > offsets() {
            return return_t< array< int_t, 4 > >{{{-1, 1, 0, 0}, {-1, -1, 1, 0}, {0, 1, 0, 0}, {0, -1, 0, 0}}};
        }
    };

    template <>
    template <>
    template <>
    struct from< edges >::to< edges >::with_color< static_int< 2 > > {

        template < typename ValueType >
        using return_t = typename return_type< from< edges >::to< edges >, ValueType >::type;

        /*
         * neighbors order
         *
         *   __1___
         *   \    /\
         *    0  /  2
         *     \/_3__\
         */
        GT_FUNCTION
        constexpr static return_t< array< int_t, 4 > > offsets() {
            return return_t< array< int_t, 4 > >{{{0, -2, 0, 0}, {0, -1, 0, 0}, {0, -2, 1, 0}, {1, -1, 0, 0}}};
        }
    };

    template <>
    template <>
    template <>
    struct from< cells >::to< edges >::with_color< static_int< 1 > > {

        template < typename ValueType = int_t >
        using return_t = typename return_type< from< cells >::to< edges >, ValueType >::type;

        /*
         * neighbors order
         *
         *        /\
         *       1  2
         *      /_0__\
         */
        GT_FUNCTION
        constexpr static return_t< array< int_t, 4 > > offsets() {
            return return_t< array< int_t, 4 > >{{{1, 0, 0, 0}, {0, 1, 0, 0}, {0, -1, 1, 0}}};
        }
    };

    template <>
    template <>
    template <>
    struct from< cells >::to< edges >::with_color< static_int< 0 > > {

        template < typename ValueType >
        using return_t = typename return_type< from< cells >::to< edges >, ValueType >::type;

        /*
         * neighbors order
         *
         *   __0___
         *   \    /
         *    2  1
         *     \/
         */
        GT_FUNCTION
        constexpr static return_t< array< int_t, 4 > > offsets() {
            return return_t< array< int_t, 4 > >{{{0, 1, 0, 0}, {0, 2, 0, 0}, {0, 0, 0, 0}}};
        }
    };

    template <>
    template <>
    template <>
    struct from< cells >::to< vertexes >::with_color< static_int< 0 > > {

        template < typename ValueType >
        using return_t = typename return_type< from< cells >::to< vertexes >, ValueType >::type;

        /*
         * neighbors order
         *
         *  1______2
         *   \    /
         *    \  /
         *     \/
         *     0
         */
        GT_FUNCTION
        constexpr static return_t< array< int_t, 4 > > offsets() {
            return return_t< array< int_t, 4 > >{{{1, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 1, 0}}};
        }
    };

    template <>
    template <>
    template <>
    struct from< cells >::to< vertexes >::with_color< static_int< 1 > > {

        template < typename ValueType >
        using return_t = typename return_type< from< cells >::to< vertexes >, ValueType >::type;

        /*
         * neighbors order
         *
         *      0
         *      /\
         *     /  \
         *    /____\
         *   2      1
         */
        GT_FUNCTION
        constexpr static return_t< array< int_t, 4 > > offsets() {
            return return_t< array< int_t, 4 > >{{{0, -1, 1, 0}, {1, -1, 1, 0}, {1, -1, 0, 0}}};
        }
    };

    template <>
    template <>
    template <>
    struct from< edges >::to< cells >::with_color< static_int< 0 > > {

        template < typename ValueType >
        using return_t = typename return_type< from< edges >::to< cells >, ValueType >::type;

        /*
         * neighbors order
         *
         *        ______
         *       /\  1 /
         *      /0 \  /
         *     /____\/
         */
        GT_FUNCTION
        constexpr static return_t< array< int_t, 4 > > offsets() {
            return return_t< array< int_t, 4 > >{{{0, 1, -1, 0}, {0, 0, 0, 0}}};
        }
    };

    template <>
    template <>
    template <>
    struct from< edges >::to< cells >::with_color< static_int< 1 > > {

        template < typename ValueType >
        using return_t = typename return_type< from< edges >::to< cells >, ValueType >::type;

        /*
         * neighbors order
         *
         *      /\
         *     / 0\
         *    /____\
         *    \    /
         *     \1 /
         *      \/
         */
        GT_FUNCTION
        constexpr static return_t< array< int_t, 4 > > offsets() {
            return return_t< array< int_t, 4 > >{{{-1, 0, 0, 0}, {0, -1, 0, 0}}};
        }
    };

    template <>
    template <>
    template <>
    struct from< edges >::to< cells >::with_color< static_int< 2 > > {

        template < typename ValueType >
        using return_t = typename return_type< from< edges >::to< cells >, ValueType >::type;

        /*
         * neighbors order
         *
         *   ______
         *   \ 1  /\
         *    \  / 0\
         *     \/____\
         */
        GT_FUNCTION
        constexpr static return_t< array< int_t, 4 > > offsets() {
            return return_t< array< int_t, 4 > >{{{0, -1, 0, 0}, {0, -2, 0, 0}}};
        }
    };

    template <>
    template <>
    template <>
    struct from< edges >::to< vertexes >::with_color< static_int< 0 > > {

        template < typename ValueType >
        using return_t = typename return_type< from< edges >::to< vertexes >, ValueType >::type;

        /*
         * neighbors order
         *
         *       1______
         *       /\    /
         *      /  \  /
         *     /____\/
         *           0
         */
        GT_FUNCTION
        constexpr static return_t< array< int_t, 4 > > offsets() {
            return return_t< array< int_t, 4 > >{{{1, 0, 0, 0}, {0, 0, 0, 0}}};
        }
    };

    template <>
    template <>
    template <>
    struct from< edges >::to< vertexes >::with_color< static_int< 1 > > {

        template < typename ValueType >
        using return_t = typename return_type< from< edges >::to< vertexes >, ValueType >::type;

        /*
         * neighbors order
         *
         *
         *      /\
         *     /  \
         *   0/____\1
         *    \    /
         *     \  /
         *      \/
         *
         */
        GT_FUNCTION
        constexpr static return_t< array< int_t, 4 > > offsets() {
            return return_t< array< int_t, 4 > >{{{0, -1, 0, 0}, {0, -1, 1, 0}}};
        }
    };

    template <>
    template <>
    template <>
    struct from< edges >::to< vertexes >::with_color< static_int< 2 > > {

        template < typename ValueType >
        using return_t = typename return_type< from< edges >::to< vertexes >, ValueType >::type;

        /*
         * neighbors order
         *
         *   ______0
         *   \    /\
         *    \  /  \
         *     \/____\
         *     1
         */
        GT_FUNCTION
        constexpr static return_t< array< int_t, 4 > > offsets() {
            return return_t< array< int_t, 4 > >{{{0, -2, 1, 0}, {1, -2, 0, 0}}};
        }
    };

    template <>
    template <>
    template <>
    struct from< vertexes >::to< cells >::with_color< static_int< 0 > > {

        template < typename ValueType >
        using return_t = typename return_type< from< vertexes >::to< cells >, ValueType >::type;

        /*
         * neighbors order
         *
         *          ______
         *         /\ 1  /\
         *        /0 \  / 2\
         *       /____\/____\
         *       \ 5  /\ 3  /
         *        \  /4 \  /
         *         \/____\/
         *
         */
        GT_FUNCTION
        constexpr static return_t< array< int_t, 4 > > offsets() {
            return return_t< array< int_t, 4 > >{
                {{-1, 1, -1, 0}, {-1, 0, 0, 0}, {-1, 1, 0, 0}, {0, 0, 0, 0}, {0, 1, -1, 0}, {0, 0, -1, 0}}};
        }
    };

    template <>
    template <>
    template <>
    struct from< vertexes >::to< edges >::with_color< static_int< 0 > > {

        template < typename ValueType >
        using return_t = typename return_type< from< vertexes >::to< edges >, ValueType >::type;

        /*
         * neighbors order
         *
         *          ______
         *         /\    /\
         *        /  1  2  \
         *       /__0_\/__3_\
         *       \    /\    /
         *        \  5  4  /
         *         \/____\/
         *
         */
        GT_FUNCTION
        constexpr static return_t< array< int_t, 4 > > offsets() {
            return return_t< array< int_t, 4 > >{
                {{0, 1, -1, 0}, {-1, 0, 0, 0}, {-1, 2, 0, 0}, {0, 1, 0, 0}, {0, 0, 0, 0}, {0, 2, -1, 0}}};
        }
    };

    template < typename SrcLocation, typename DesLocation >
    struct connectivity;

    template < typename DestLocation >
    struct connectivity< cells, DestLocation > {
        GT_FUNCTION
        static typename return_type< typename from< cells >::template to< DestLocation >, array< int_t, 4 > >::type
        offsets(const uint_t color) {
            if (color == 0) {
                return from< cells >::template to< DestLocation >::template with_color< static_int< 0 > >::offsets();
            } else if (color == 1) {
                return from< cells >::template to< DestLocation >::template with_color< static_int< 1 > >::offsets();
            } else {
                assert(true);
            }
        }
    };

    template < typename DestLocation >
    struct connectivity< edges, DestLocation > {
        GT_FUNCTION
        static typename return_type< typename from< edges >::template to< DestLocation >, array< int_t, 4 > >::type
        offsets(const uint_t color) {
            if (color == 0) {
                return from< edges >::template to< DestLocation >::template with_color< static_int< 0 > >::offsets();
            } else if (color == 1) {
                return from< edges >::template to< DestLocation >::template with_color< static_int< 1 > >::offsets();
            } else if (color == 2) {
                return from< edges >::template to< DestLocation >::template with_color< static_int< 2 > >::offsets();
            } else {
                assert(true);
            }
        }
    };

    template < typename DestLocation >
    struct connectivity< vertexes, DestLocation > {
        GT_FUNCTION
        static typename return_type< typename from< vertexes >::template to< DestLocation >, array< int_t, 4 > >::type
        offsets(const uint_t color) {
            if (color == 0) {
                return from< vertexes >::template to< DestLocation >::template with_color< static_int< 0 > >::offsets();
            } else {
                assert(true);
            }
        }
    };

    template < typename Location1, typename Location2, uint_t Color >
    struct connectivity_indexes {
        template < typename ValueType >
        using return_t = typename return_type< typename from< Location1 >::template to< Location2 >, ValueType >::type;

        static const size_t n_neighbors = return_t< array< uint_t, 4 > >::n_dimensions;

        /**
          * function to extract the 4 indexes of all neighbours of current position, when the neighbours are not in the
          * same location as the location type of the iteration space.
          * @return an array (over neighbours) of an array (indices of position).
          *     Dimension of the outer array depends on the number of neighbours of the location type
          * @i indexes of current position in the iteration space
          */
        template < typename Grid >
        GT_FUNCTION static return_t< uint_t > get_index(Grid const &grid, array< uint_t, 3 > const &i) {

            //Note: offsets have to be extracted here as a constexpr object instead of passed inline to the apply fn
            // Otherwise constexpr of the array is lost
            constexpr const auto offsets =
                from< Location1 >::template to< Location2 >::template with_color< static_int< Color > >::offsets();
            using seq = gridtools::apply_gt_integer_sequence<
                typename gridtools::make_gt_integer_sequence< int, n_neighbors >::type >;
            return seq::template apply< return_t< uint_t >,
                get_connectivity_index< Location2, Grid, Color >::template get_element >(grid, i, offsets);
        }
    };

    template < typename Location, uint_t Color >
    struct connectivity_indexes<Location, Location, Color> {
        template < typename ValueType >
        using return_t = typename return_type< typename from< Location >::template to< Location >, ValueType >::type;

        static const size_t n_neighbors = return_t< array< uint_t, 4 > >::n_dimensions;

        /**
          * function to extract the 4 indexes of all neighbours of current position, when the neighbours are in the same
          * location as the location type of the iteration space.
          * @return an array (over neighbours) of an array (indices of position).
          *     Dimension of the outer array depends on the number of neighbours of the location type
          * @i indexes of current position in the iteration space
          */
        GT_FUNCTION
        static return_t< array< uint_t, 4 > > get_index(array< uint_t, 3 > const &i) {

            //Note: offsets have to be extracted here as a constexpr object instead of passed inline to the apply fn
            // Otherwise constexpr of the array is lost
            constexpr const auto offsets =
                from< Location >::template to< Location >::template with_color< static_int< Color > >::offsets();
            using seq = gridtools::apply_gt_integer_sequence<
                typename gridtools::make_gt_integer_sequence< int, n_neighbors >::type >;
            return seq::template apply< return_t< array< uint_t, 4 > >,
                get_connectivity_offset< Color >::template get_element >(i, offsets);
        }
    };

    /**
    */
    template < typename Backend >
    class icosahedral_topology : public clonable_to_gpu< icosahedral_topology< Backend > > {
      public:
        using cells = location_type< 0, 2 >;
        using edges = location_type< 1, 3 >;
        using vertexes = location_type< 2, 1 >;

        template < typename LocationType >
        using meta_storage_t = typename Backend::template storage_info_t< LocationType >;

        template < typename LocationType, typename ValueType >
        using storage_t = typename Backend::template storage_t< LocationType, ValueType >;

        const gridtools::array< uint_t, 2 > m_dims; // Sizes as cells in a multi-dimensional Cell array

        using grid_meta_storages_t =
            boost::fusion::vector3< meta_storage_t< cells >, meta_storage_t< edges >, meta_storage_t< vertexes > >;

        grid_meta_storages_t m_virtual_storages;

      public:
        using n_locations = static_uint< boost::mpl::size< grid_meta_storages_t >::value >;
        template < typename LocationType >
        GT_FUNCTION uint_t size(LocationType location) {
            return boost::fusion::at_c< LocationType::value >(m_virtual_storages).size();
        }

        icosahedral_topology() = delete;

      public:
        template < typename... UInt >
        GT_FUNCTION icosahedral_topology(uint_t first_, uint_t second_, UInt... dims)
            : m_dims{second_, first_},
              m_virtual_storages(meta_storage_t< cells >(array< uint_t, meta_storage_t< cells >::space_dimensions >{
                                     first_, cells::n_colors::value, second_, dims...}),
                  meta_storage_t< edges >(array< uint_t, meta_storage_t< edges >::space_dimensions >{
                      first_, edges::n_colors::value, second_, dims...}),
                  // here we assume by convention that the dual grid (vertexes) have one more grid point
                  meta_storage_t< vertexes >(array< uint_t, meta_storage_t< vertexes >::space_dimensions >{
                      first_, vertexes::n_colors::value, second_ + 1, dims...})) {}

        __device__ icosahedral_topology(icosahedral_topology const &other)
            : m_dims(other.m_dims), m_virtual_storages(boost::fusion::at_c< cells::value >(other.m_virtual_storages),
                                        boost::fusion::at_c< edges::value >(other.m_virtual_storages),
                                        boost::fusion::at_c< vertexes::value >(other.m_virtual_storages)) {}

        GT_FUNCTION
        grid_meta_storages_t const &virtual_storages() const { return m_virtual_storages; }

        // TODOMEETING move semantic
        template < typename LocationType, typename ValueType >
        GT_FUNCTION storage_t< LocationType, double > make_storage(char const *name) const {
            return storage_t< LocationType, ValueType >(
                boost::fusion::at_c< LocationType::value >(m_virtual_storages), name);
        }

        template < typename LocationType >
        GT_FUNCTION array< int_t, 4 > ll_indices(array< int_t, 3 > const &i, LocationType) const {
            auto out = array< int_t, 4 >{i[0],
                i[1] % static_cast< int_t >(LocationType::n_colors::value),
                i[1] / static_cast< int >(LocationType::n_colors::value),
                i[2]};
            return array< int_t, 4 >{i[0],
                i[1] % static_cast< int_t >(LocationType::n_colors::value),
                i[1] / static_cast< int >(LocationType::n_colors::value),
                i[2]};
        }

        template < typename LocationType >
        GT_FUNCTION int_t ll_offset(array< uint_t, 4 > const &i, LocationType) const {
            return boost::fusion::at_c< LocationType::value >(m_virtual_storages).index(i);
        }

        // methods returning the neighbors. Specializations according to the location type
        // needed a way to implement static double dispatch
        template < typename Location1, typename Location2, typename Color >
        GT_FUNCTION typename return_type< typename from< Location1 >::template to< Location2 >, uint_t >::type const
                ll_map_index(Location1, Location2, Color, array< uint_t, 3 > const &i) const {
            return connectivity_indexes< Location1, Location2, Color::value >::get_index(*this, i);
        }

        // methods returning the neighbors. Specializations according to the location type
        // needed a way to implement static double dispatch
        template < typename Location, typename Color >
        GT_FUNCTION static
            typename return_type< typename from< Location >::template to< Location >, array< uint_t, 4 > >::type const
                ll_map_index(Location, Location, Color, array< uint_t, 3 > const &i) {
            return connectivity_indexes< Location, Location, Color::value >::get_index(i);
        }

        template < typename Location2 > // Works for cells or edges with same code
        GT_FUNCTION
            typename return_type< typename from< cells >::template to< Location2 >, array< uint_t, 4 > >::type
                neighbors_indices_3(array< uint_t, 4 > const &i, cells, Location2) const {
            switch (i[1] % cells::n_colors::value) {
            case 0:
                return ll_map_index(cells(), Location2(), static_int< 0 >(), {i[0], i[2], i[3]});
            case 1:
                return ll_map_index(cells(), Location2(), static_int< 1 >(), {i[0], i[2], i[3]});
            default:
                GTASSERT(false);
                return typename return_type< typename from< cells >::template to< Location2 >,
                    array< uint_t, 4 > >::type();
            }
        }

        template < typename Location2 > // Works for cells or edges with same code
        GT_FUNCTION
            typename return_type< typename from< edges >::template to< Location2 >, array< uint_t, 4 > >::type
                neighbors_indices_3(array< uint_t, 4 > const &i, edges, Location2) const {
            switch (i[1] % edges::n_colors::value) {
            case 0:
                return ll_map_index(edges(), Location2(), static_int< 0 >(), {i[0], i[2], i[3]});
            // return edge2edges_ll_p0_indices({i[0], i[2]});
            case 1:
                return ll_map_index(edges(), Location2(), static_int< 1 >(), {i[0], i[2], i[3]});
            // return edge2edges_ll_p1_indices({i[0], i[2]});
            case 2:
                return ll_map_index(edges(), Location2(), static_int< 2 >(), {i[0], i[2], i[3]});
            // return edge2edges_ll_p2_indices({i[0], i[2]});
            default:
                GTASSERT(false);
                return typename return_type< typename from< edges >::template to< Location2 >,
                    array< uint_t, 4 > >::type();
            }
        }

        template < typename Location2 > // Works for cells or edges with same code
        GT_FUNCTION
            typename return_type< typename from< vertexes >::template to< Location2 >, array< uint_t, 4 > >::type
                neighbors_indices_3(array< uint_t, 4 > const &i, vertexes, Location2) const {
            return ll_map_index(vertexes(), Location2(), static_int< 0 >(), {i[0], i[2], i[3]});
        }
    };

    template < typename T >
    struct is_grid_topology : boost::mpl::false_ {};

    template < typename Backend >
    struct is_grid_topology< icosahedral_topology< Backend > > : boost::mpl::true_ {};

} // namespace gridtools
