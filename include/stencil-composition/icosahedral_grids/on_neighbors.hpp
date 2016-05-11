#pragma once

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
    template < typename ValueType, typename DstLocationType, typename ReductionFunction, typename MapFunction >
    class on_neighbors_impl {
        using map_function = MapFunction;
        using reduction_function = ReductionFunction;
        using dst_location_type = DstLocationType;
        using value_type = ValueType;

        const reduction_function m_reduction;
        const map_function m_map;
        const value_type m_value;

      public:
        GT_FUNCTION
        on_neighbors_impl(const reduction_function l, map_function a, value_type v)
            : m_reduction(l), m_map(a), m_value(v) {}

        GT_FUNCTION
        value_type value() const { return m_value; }

        GT_FUNCTION
        reduction_function reduction() const { return m_reduction; }

        GT_FUNCTION
        map_function map() const { return m_map; }

        GT_FUNCTION
        on_neighbors_impl(on_neighbors_impl const &other)
            : m_reduction(other.m_reduction), m_map(other.m_map), m_value(other.m_value) {}

        GT_FUNCTION
        dst_location_type location() const { return dst_location_type(); }
    };

    template < typename ValueType, typename DstLocationType, typename ReductionFunction, uint_t I, typename L, int_t R >
    class on_neighbors_impl< ValueType,
        DstLocationType,
        ReductionFunction,
        accessor< I, enumtype::in, L, extent< R > > > {
        using map_function = accessor< I, enumtype::in, L, extent< R > >;
        using reduction_function = ReductionFunction;
        using dst_location_type = DstLocationType;
        using value_type = ValueType;

        const reduction_function m_reduction;
        const map_function m_map;
        const value_type m_value;

      public:
        GT_FUNCTION
        on_neighbors_impl(const reduction_function l, map_function a, value_type v)
            : m_reduction(l), m_map(a), m_value(v) {}

        // copy ctor from an accessor with different ID
        template < ushort_t OtherID >
        GT_FUNCTION constexpr explicit on_neighbors_impl(const on_neighbors_impl< ValueType,
            DstLocationType,
            ReductionFunction,
            accessor< OtherID, enumtype::in, L, extent< R > > > &other)
            : m_reduction(other.reduction()), m_map(other.map()), m_value(other.value()) {}

        GT_FUNCTION
        value_type value() const { return m_value; }

        GT_FUNCTION
        reduction_function reduction() const { return m_reduction; }

        GT_FUNCTION
        map_function map() const { return m_map; }

        GT_FUNCTION
        on_neighbors_impl(on_neighbors_impl const &other)
            : m_reduction(other.m_reduction), m_map(other.m_map), m_value(other.m_value) {}

        GT_FUNCTION
        dst_location_type location() const { return dst_location_type(); }
    };

    template < typename Reduction, typename ValueType, typename Map >
    GT_FUNCTION on_neighbors_impl< ValueType, typename Map::location_type, Reduction, Map > reduce_on_something(
        Reduction function, ValueType initial, Map mapf) {
        return on_neighbors_impl< ValueType, typename Map::location_type, Reduction, Map >(function, mapf, initial);
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
