#pragma once
#include "../../common/tuple.hpp"

namespace gridtools {

    /**
       Map function that uses compile time (stateless) accessors to be
       evaluated later. Another version would have the Arguments to be
       a fusion vector (for instance), so that each argument can carry
       additional state, like a constant value.
     */
    template < typename MapF, typename LocationType, typename... Arguments >
    struct map_function {
        using location_type = LocationType;
        using argument_types = std::tuple< Arguments... >;
        using function_type = MapF;

        const function_type m_function;
        argument_types m_arguments;

        GT_FUNCTION
        map_function(function_type f, Arguments... args) : m_function(f), m_arguments(args...) {}

        template < uint_t I >
        GT_FUNCTION typename std::tuple_element< I, argument_types >::type const &argument() const {
            return std::get< I >(m_arguments);
        }

        GT_FUNCTION
        location_type location() const { return location_type(); }

        GT_FUNCTION
        function_type function() const { return m_function; }
    };

    /**
    initial version of this that should check if all args have the same location type
    */
    template < typename Arg0, typename... Args >
    struct get_location_type_of {
        using type = typename Arg0::location_type;
    };

    template < typename MapF, typename... Args >
    map_function< MapF, typename get_location_type_of< Args... >::type, Args... > map(MapF const &f, Args... args) {
        return map_function< MapF, typename get_location_type_of< Args... >::type, Args... >(f, args...);
    }

    template < typename T >
    struct identity {
        GT_FUNCTION
        T operator()(T v) const { return v; }
    };

    /**
       This struct is the one holding the function to apply when iterating
       on neighbors
     */
    template < typename ValueType, typename DstLocationType, typename ReductionFunction, typename ... MapFunction >
    class on_neighbors_impl {
        using maps_t = tuple<MapFunction...>;
        using reduction_function = ReductionFunction;
        using dst_location_type = DstLocationType;
        using value_type = ValueType;

        const reduction_function m_reduction;
        const maps_t m_maps;
        const value_type m_value;

      public:
        GT_FUNCTION
        constexpr on_neighbors_impl(const reduction_function l, value_type v, MapFunction ... a)
            : m_reduction(l), m_value(v), m_maps(a...) {}

        GT_FUNCTION
        value_type value() const { return m_value; }

        GT_FUNCTION
        reduction_function reduction() const { return m_reduction; }


        template<ushort_t idx>
        GT_FUNCTION
        constexpr typename maps_t::template get_elem<idx>::type map() const { return m_maps.template get<idx>(); }

        GT_FUNCTION
        on_neighbors_impl(on_neighbors_impl const &other)
            : m_reduction(other.m_reduction), m_value(other.m_value), m_maps(other.m_maps) {}

        GT_FUNCTION
        dst_location_type location() const { return dst_location_type(); }
    };

    template < typename Reduction, typename ValueType, typename Map >
    GT_FUNCTION on_neighbors_impl< ValueType, typename Map::location_type, Reduction, Map > reduce_on_something(
        Reduction function, ValueType initial, Map mapf) {
        return on_neighbors_impl< ValueType, typename Map::location_type, Reduction, Map >(function, initial, mapf);
    }

    template < typename Reduction, typename ValueType, typename Map >
    GT_FUNCTION on_neighbors_impl< ValueType, typename Map::location_type, Reduction, Map > on_edges(
        Reduction function, ValueType initial, Map mapf) {
        static_assert(Map::location_type::value == 1,
            "The map function (for a nested call) provided to 'on_edges' is not on edges");
        return reduce_on_something(function, initial, mapf);
    }

    template < typename Reduction, typename ValueType, typename Map >
    GT_FUNCTION on_neighbors_impl< ValueType, typename Map::location_type, Reduction, Map > on_cells(
        Reduction function, ValueType initial, Map mapf) {
        GRIDTOOLS_STATIC_ASSERT(Map::location_type::value == 0,
            "The map function (for a nested call) provided to 'on_cellss' is not on cells");
        return reduce_on_something(function, initial, mapf);
    }

    template < typename Reduction, typename ValueType, typename Map >
    GT_FUNCTION on_neighbors_impl< ValueType, typename Map::location_type, Reduction, Map > on_vertexes(
        Reduction function, ValueType initial, Map mapf) {
        GRIDTOOLS_STATIC_ASSERT(Map::location_type::value == 2,
            "The map function (for a nested call) provided to 'on_vertexes' is not on edges");
        return reduce_on_something(function, initial, mapf);
    }

    template < typename OnNeighbors, typename RemapAccessor >
    struct remap_on_neighbors;

    template < typename ValueType,
        typename DstLocationType,
        typename ReductionFunction,
        uint_t Index,
        enumtype::intend Intend,
        typename LocationType,
        int_t R,
        uint_t OtherIndex >
    struct remap_on_neighbors< on_neighbors_impl< ValueType,
                                   DstLocationType,
                                   ReductionFunction,
                                   accessor< Index, Intend, LocationType, extent< R > > >,
        accessor< OtherIndex, Intend, LocationType, extent< R > > > {
        typedef on_neighbors_impl< ValueType,
            DstLocationType,
            ReductionFunction,
            accessor< OtherIndex, Intend, LocationType, extent< R > > > type;
    };

} // namespace gridtools
