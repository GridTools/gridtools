#pragma once
#include "location_type.hpp"
#include <type_traits>
#include "common/generic_metafunctions/apply_to_sequence.hpp"
#include "common/generic_metafunctions/vector_to_set.hpp"
#include "stencil-composition/iterate_domain_impl_metafunctions.hpp"
#include "stencil-composition/total_storages.hpp"
#include "stencil-composition/iterate_domain_aux.hpp"
#include "stencil-composition/icosahedral_grids/accessor_metafunctions.hpp"
#include "stencil-composition/iterate_domain_impl.hpp"

#define _ACCESSOR_H_DEBUG_

namespace gridtools {

/**
   Map function that uses compile time (stateless) accessors to be
   evaluated later. Another version would have the Arguments to be
   a fusion vector (for instance), so that each argument can carry
   additional state, like a constant value.
 */
template <typename MapF, typename LocationType, typename ... Arguments>
struct map_function {
    using location_type = LocationType;
    using argument_types = std::tuple<Arguments...>;
    using function_type = MapF;

    const function_type m_function;
    argument_types m_arguments;

    map_function(function_type f, Arguments... args)
        : m_function(f)
        , m_arguments(args...)
    {}

    template <int I>
    typename std::tuple_element<I, argument_types>::type const&
    argument() const {
        return std::get<I>(m_arguments);
    }

    location_type location() const {return location_type();}

    function_type function() const {return m_function;}
};

/**
initial version of this that should check if all args have the same location type
*/
template <typename Arg0, typename ... Args>
struct get_location_type_of {
    using type = typename Arg0::location_type;
};

template <typename MapF, typename ... Args>
map_function<MapF, typename get_location_type_of<Args...>::type, Args...>
map(MapF const& f, Args... args) {
    return map_function<MapF, typename get_location_type_of<Args...>::type, Args...>(f, args...);
}

template <typename T>
struct identity {
    T operator()(T v) const
    {
        return v;
    }
};

/**
   This struct is the one holding the function to apply when iterating
   on neighbors
 */
template <typename ValueType
          , typename DstLocationType
          , typename ReductionFunction
          , typename MapFunction
          >
class on_neighbors_impl {
    using map_function = MapFunction;
    using reduction_function = ReductionFunction;
    using dst_location_type = DstLocationType;
    using value_type = ValueType;

    const reduction_function m_reduction;
    const map_function m_map;
    const value_type m_value;

public:
    on_neighbors_impl(const reduction_function l, map_function a, value_type v)
        : m_reduction(l)
        , m_map(a)
        , m_value(v)
    {}

    value_type value() const {return m_value;}

    reduction_function reduction() const {return m_reduction;}

    map_function map() const {return m_map;}

    on_neighbors_impl(on_neighbors_impl const& other)
        : m_reduction(other.m_reduction)
        , m_map(other.m_map)
        , m_value(other.m_value)
    {}

    dst_location_type location() const
    {
        return dst_location_type();
    }
};

template <typename Reduction
          , typename ValueType
          , typename Map
          >
on_neighbors_impl<ValueType
                  , typename Map::location_type
                  , Reduction
                  , Map
                  >
reduce_on_something(Reduction function
                    , ValueType initial
                    , Map mapf)
{
    return on_neighbors_impl<ValueType
                             , typename Map::location_type
                             , Reduction
                             , Map
                             >(function, mapf, initial);
}

template <typename Reduction
          , typename ValueType
          , typename Map
          >
on_neighbors_impl<ValueType
                  , typename Map::location_type
                  , Reduction
                  , Map
                  >
on_edges(Reduction function
                , ValueType initial
                , Map mapf)
{
    static_assert(Map::location_type::value==1,
                  "The map function (for a nested call) provided to 'on_edges' is not on edges");
    return reduce_on_something(function, initial, mapf);
}

template <typename Reduction
          , typename ValueType
          , typename Map
          >
on_neighbors_impl<ValueType
                  , typename Map::location_type
                  , Reduction
                  , Map
                  >
on_cells(Reduction function
                , ValueType initial
                , Map mapf)
{
    static_assert(Map::location_type::value==0,
                  "The map function (for a nested call) provided to 'on_cellss' is not on cells");
    return reduce_on_something(function, initial, mapf);
}

template <typename Reduction
          , typename ValueType
          , typename Map
          >
on_neighbors_impl<ValueType
                  , typename Map::location_type
                  , Reduction
                  , Map
                  >
on_vertexes(Reduction function
                , ValueType initial
                , Map mapf)
{
    static_assert(Map::location_type::value==2,
                  "The map function (for a nested call) provided to 'on_vertexes' is not on edges");
    return reduce_on_something(function, initial, mapf);
}



//TODO move this to the appropiate file
template<typename EsfSequence>
struct extract_location_type
{
    GRIDTOOLS_STATIC_ASSERT((is_sequence_of<EsfSequence, is_esf_descriptor>::value), "Error: wrong type");
    typedef typename apply_to_sequence<EsfSequence, esf_get_location_type>::type location_type_seq_t;
    typedef typename vector_to_set<location_type_seq_t>::type location_type_set_t;

    GRIDTOOLS_STATIC_ASSERT((boost::mpl::size<location_type_set_t>::value == 1),
        "Error: multiple ESFs were used with different location types."
        " Currently all esf must be specified on the same location type. "
        "Future releases will relax this restriction"
    );
    typedef typename boost::mpl::front<location_type_set_t>::type type;

};

/**
   This class is basically the iterate domain. It contains the
   ways to access data and the implementation of iterating on neighbors.
 */
//template <typename PlcVector, typename GridType, typename LocationType>
template <typename IterateDomainImpl>
struct iterate_domain {
    typedef typename iterate_domain_impl_arguments<IterateDomainImpl>::type iterate_domain_arguments_t;
    typedef typename iterate_domain_arguments_t::local_domain_t local_domain_t;

    typedef typename iterate_domain_arguments_t::backend_id_t backend_id_t;
    typedef typename iterate_domain_arguments_t::grid_t::grid_topology_t grid_topology_t;
    typedef typename iterate_domain_arguments_t::esf_sequence_t esf_sequence_t;
    typedef typename extract_location_type<esf_sequence_t>::type location_type_t;

    typedef typename local_domain_t::esf_args esf_args_t;

    typedef typename backend_traits_from_id< backend_id_t::value >::
            template select_iterate_domain_cache<iterate_domain_arguments_t>::type iterate_domain_cache_t;

    typedef typename iterate_domain_cache_t::ij_caches_map_t ij_caches_map_t;

    GRIDTOOLS_STATIC_ASSERT((is_local_domain<local_domain_t>::value), "Internal Error: wrong type");
    typedef typename boost::remove_pointer<
        typename boost::mpl::at_c<
            typename local_domain_t::mpl_storages, 0>::type
        >::type::value_type value_type;

    typedef typename local_domain_t::storage_metadata_map metadata_map_t;
    typedef typename local_domain_t::actual_args_type actual_args_type;

    //the number of different storage metadatas used in the current functor
    static const uint_t N_META_STORAGES=boost::mpl::size<metadata_map_t>::value;
    //the number of storages  used in the current functor
    static const uint_t N_STORAGES=boost::mpl::size<actual_args_type>::value;
    //the total number of snapshot (one or several per storage)
    static const uint_t N_DATA_POINTERS=total_storages<
        actual_args_type,
        boost::mpl::size<typename local_domain_t::mpl_storages>::type::value >::value;

    typedef array<void* RESTRICT, N_DATA_POINTERS> data_pointer_array_t;
    typedef strides_cached<N_META_STORAGES-1, typename local_domain_t::storage_metadata_vector_t> strides_cached_t;

    /**@brief local class instead of using the inline (cond)?a:b syntax, because in the latter both branches get compiled (generating sometimes a compile-time overflow) */
    template <bool condition, typename LocalD, typename Accessor>
    struct current_storage;

    template < typename LocalD, typename Accessor>
    struct current_storage<true, LocalD, Accessor>{
        static const uint_t value=0;
    };

    template < typename LocalD, typename Accessor>
    struct current_storage<false, LocalD, Accessor>{
        static const uint_t value=(total_storages< typename LocalD::local_args_type, Accessor::index_type::value >::value);
    };

    /**
     * metafunction that computes the return type of all operator() of an accessor
     */
    template<typename Accessor>
    struct accessor_return_type
    {
        typedef typename ::gridtools::accessor_return_type<Accessor, iterate_domain_arguments_t>::type type;
    };

private:
    GRIDTOOLS_STATIC_ASSERT((N_META_STORAGES <= grid_topology_t::n_locations::value),"We can not have more meta storages"
                            "than location types. Data fields for other grids are not yet supported");
    local_domain_t const& m_local_domain;
    grid_topology_t const& m_grid_topology;
    //TODOMEETING do we need m_index?
    array<int_t,N_META_STORAGES> m_index;

    array<uint_t, 4> m_grid_position;
public:

    /**@brief constructor of the iterate_domain struct

       It assigns the storage pointers to the first elements of
       the data fields (for all the data_fields present in the
       current evaluation), and the indexes to access the data
       fields (one index per storage instance, so that one index
       might be shared among several data fileds)
    */
    GT_FUNCTION
    iterate_domain(local_domain_t const& local_domain_, grid_topology_t const& grid_topology)
        : m_local_domain(local_domain_), m_grid_topology(grid_topology) {}

    /**
       @brief returns the array of pointers to the raw data
    */
    GT_FUNCTION
    data_pointer_array_t const& RESTRICT data_pointer() const
    {
        return static_cast<IterateDomainImpl const *>(this)->data_pointer_impl();
    }

    /**
       @brief returns the array of pointers to the raw data
    */
    GT_FUNCTION
    data_pointer_array_t& RESTRICT data_pointer()
    {
        return static_cast<IterateDomainImpl*>(this)->data_pointer_impl();
    }

    /**
       @brief returns the strides as const reference
    */
    GT_FUNCTION
    strides_cached_t const & RESTRICT strides() const
    {
        return static_cast<IterateDomainImpl const *>(this)->strides_impl();
    }

    /**
       @brief returns the strides as const reference
    */
    GT_FUNCTION
    strides_cached_t & RESTRICT strides()
    {
        return static_cast<IterateDomainImpl*>(this)->strides_impl();
    }


    /** This functon set the addresses of the data values  before the computation
        begins.

        The EU stands for ExecutionUnit (thich may be a thread or a group of
        threasd. There are potentially two ids, one over i and one over j, since
        our execution model is parallel on (i,j). Defaulted to 1.
    */
    template<typename BackendType>
    GT_FUNCTION
    void assign_storage_pointers(){
        const uint_t EU_id_i = BackendType::processing_element_i();
        const uint_t EU_id_j = BackendType::processing_element_j();
        for_each<typename reversed_range< int_t, 0, N_STORAGES >::type > (
            assign_storage_functor<
                BackendType,
                data_pointer_array_t,
                typename local_domain_t::local_args_type,
                typename local_domain_t::local_metadata_type,
                metadata_map_t
            >(data_pointer(), m_local_domain.m_local_args, m_local_domain.m_local_metadata,  EU_id_i, EU_id_j));
    }

    /**
       @brief recursively assignes all the strides

       copies them from the
       local_domain.m_local_metadata vector, and stores them into an instance of the
       \ref strides_cached class.
     */
    template<typename BackendType, typename Strides>
    GT_FUNCTION
    void assign_stride_pointers(){
        GRIDTOOLS_STATIC_ASSERT((is_strides_cached<Strides>::value), "internal error type");
        for_each< metadata_map_t > (
            assign_strides_functor<
            BackendType,
            Strides,
            typename boost::fusion::result_of::as_vector
            <typename local_domain_t::local_metadata_type>::type
            >(strides(), m_local_domain.m_local_metadata));
    }

    /**@brief method for initializing the index */
    template <ushort_t Coordinate>
    GT_FUNCTION
    void initialize(uint_t const initial_pos=0, uint_t const block=0)
    {
        for_each< metadata_map_t > (
            initialize_index_functor<
            Coordinate,
            strides_cached_t,
            typename boost::fusion::result_of::as_vector
            <typename local_domain_t::local_metadata_type>::type
            >(strides(), boost::fusion::as_vector(m_local_domain.m_local_metadata), initial_pos, block, &m_index[0]));
        static_cast<IterateDomainImpl*>(this)->template initialize_impl<Coordinate>();

        m_grid_position[Coordinate] = initial_pos;
    }

    void set_data_pointer(data_pointer_array_t* RESTRICT data_pointer)
    {
        static_cast<IterateDomainImpl*>(this)->template set_data_pointer_impl(data_pointer);
    }

    void set_strides_pointer(strides_cached_t* RESTRICT strides)
    {
        static_cast<IterateDomainImpl*>(this)->template set_strides_pointer_impl(strides);
    }


    /**@brief method for incrementing by 1 the index when moving forward along the given direction
       \tparam Coordinate dimension being incremented
       \tparam Execution the policy for the increment (e.g. forward/backward)
     */
    template <ushort_t Coordinate, typename Execution>
    GT_FUNCTION
    void increment()
    {
        for_each< metadata_map_t > (
            increment_index_functor<
            Coordinate,
            strides_cached_t,
            typename boost::fusion::result_of::as_vector
            <typename local_domain_t::local_metadata_type>::type
            >(boost::fusion::as_vector(m_local_domain.m_local_metadata),
#ifdef __CUDACC__ //stupid nvcc
              boost::is_same<Execution, static_int<1> >::type::value? 1 : -1
#else
              Execution::value
#endif
              , &m_index[0], strides())
            );
        static_cast<IterateDomainImpl*>(this)->template increment_impl<Coordinate, Execution>();
        m_grid_position[Coordinate] += Execution::value;
    }

    /**@brief method for incrementing the index when moving forward along the given direction

       \param steps_ the increment
       \tparam Coordinate dimension being incremented
     */
    template <ushort_t Coordinate>
    GT_FUNCTION
    void increment(int_t steps_)
    {
        for_each< metadata_map_t > (
            increment_index_functor<
                Coordinate,
                strides_cached_t,
            typename boost::fusion::result_of::as_vector
            <typename local_domain_t::local_metadata_type>::type
            >(boost::fusion::as_vector(m_local_domain.m_local_metadata), steps_, &m_index[0], strides())
        );
        static_cast<IterateDomainImpl*>(this)->template increment_impl<Coordinate>(steps_);

        m_grid_position[Coordinate] += steps_;
    }

    /**@brief getter for the index array */
    //TODO simplify this using just loops
    GT_FUNCTION
    void get_index(array<int_t, N_META_STORAGES>& index) const
    {
        set_index_recur< N_META_STORAGES-1>::set(m_index, index);
    }

    /**@brief getter for the index array */
    GT_FUNCTION
    void get_position(array<uint_t, 4>& position) const
    {
        position = m_grid_position;
    }

    /**@brief method for setting the index array */
    //TODO no need for an template parameter here
    template <typename Input>
    GT_FUNCTION
    void set_index(Input const& index)
    {
        set_index_recur< N_META_STORAGES-1>::set( index, m_index);
    }

    GT_FUNCTION
    void set_position(array<uint_t, 4> const& position)
    {
        m_grid_position = position;
    }

    template<typename Accessor>
    GT_FUNCTION
    typename accessor_return_type<Accessor>::type
    operator()(Accessor const& accessor) const {
        GRIDTOOLS_STATIC_ASSERT((is_accessor<Accessor>::value), "Using EVAL is only allowed for an accessor type");
        return get_value(accessor, (data_pointer())[current_storage<(Accessor::index_type::value==0)
                                                , local_domain_t, typename Accessor::type >::value]);
    }

    template <typename ValueType
              , typename LocationTypeT
              , typename Reduction
              , typename MapF
              , typename ...Arg0
              >
    double operator()(on_neighbors_impl<ValueType, LocationTypeT, Reduction, map_function<MapF, LocationTypeT, Arg0...>> onneighbors) const {
        auto current_position = m_grid_position;

        const auto neighbors = grid_topology_t::neighbors_indices_3(current_position
                                                          , location_type_t()
                                                          , onneighbors.location() );
        double result = onneighbors.value();

        for (int i = 0; i<neighbors.size(); ++i) {
            result = onneighbors.reduction()( _evaluate(onneighbors.map(), neighbors[i]), result );
        }

        return result;
    }

    template <typename ValueType
              , typename LocationTypeT
              , typename Reduction
              , int I
              , typename L
              , int R
              >
    double operator()(on_neighbors_impl<ValueType, LocationTypeT, Reduction, ro_accessor<I,L,radius<R>>> onneighbors) const {
        auto current_position = m_grid_position;

        const auto neighbors = grid_topology_t::neighbors_indices_3(current_position
                                                          , location_type_t()
                                                          , onneighbors.location() );
        double result = onneighbors.value();

        for (int i = 0; i<neighbors.size(); ++i) {
            result = onneighbors.reduction()( _evaluate(onneighbors.map(), neighbors[i]), result );
        }

        return result;
    }


    /**@brief returns the value of the memory at the given address, plus the offset specified by the arg placeholder
       \param arg placeholder containing the storage ID and the offsets
       \param storage_pointer pointer to the first element of the specific data field used
    */
    template <typename Accessor, typename StoragePointer>
    GT_FUNCTION
    typename accessor_return_type<Accessor>::type
    get_value(Accessor const& accessor , StoragePointer & RESTRICT storage_pointer) const {

        //getting information about the storage
        typedef typename Accessor::index_type index_t;

#ifndef CXX11_ENABLED
        typedef typename boost::remove_reference<typename boost::remove_pointer<BOOST_TYPEOF( (boost::fusion::at
               < index_t>(m_local_domain.m_local_args)) )>::type>::type storage_type;
        storage_type* const storage_=
#else
        auto const storage_ =
#endif
            boost::fusion::at
            < index_t>(m_local_domain.m_local_args);

        GRIDTOOLS_STATIC_ASSERT((is_accessor<Accessor>::value), "Using EVAL is only allowed for an accessor type");

#ifdef CXX11_ENABLED
        using storage_type = typename std::remove_reference<decltype(*storage_)>::type;
#endif
        typename storage_type::value_type * RESTRICT real_storage_pointer=
                static_cast<typename storage_type::value_type*>(storage_pointer);

        //getting information about the metadata
        typedef typename boost::mpl::at
            <metadata_map_t, typename storage_type::meta_data_t >::type metadata_index_t;

        pointer<const typename storage_type::meta_data_t> const metadata_ = boost::fusion::at
            < metadata_index_t >(m_local_domain.m_local_metadata);
        //getting the value

        //the following assert fails when an out of bound access is observed, i.e. either one of
        //i+offset_i or j+offset_j or k+offset_k is too large.
        //Most probably this is due to you specifying a positive offset which is larger than expected,
        //or maybe you did a mistake when specifying the ranges in the placehoders definition
        assert(metadata_->size() >  metadata_->index(m_grid_position) );

        //the following assert fails when an out of bound access is observed,
        //i.e. when some offset is negative and either one of
        //i+offset_i or j+offset_j or k+offset_k is too small.
        //Most probably this is due to you specifying a negative offset which is
        //smaller than expected, or maybe you did a mistake when specifying the ranges
        //in the placehoders definition.
        // If you are running a parallel simulation another common reason for this to happen is
        // the definition of an halo region which is too small in one direction
        // std::cout<<"Storage Index: "<<Accessor::index_type::value<<" + "<<(boost::fusion::at<typename Accessor::index_type>(local_domain.local_args))->_index(arg.template n<Accessor::n_dim>())<<std::endl;
        assert( (int_t)(metadata_->index(m_grid_position)) >= 0);

        return *(real_storage_pointer+metadata_->index(m_grid_position));
    }

    template <typename Accessor, typename StoragePointer>
    GT_FUNCTION
    typename accessor_return_type<Accessor>::type
    get_raw_value(Accessor const& accessor , StoragePointer & RESTRICT storage_pointer, const uint_t offset) const {

        //getting information about the storage
        typedef typename Accessor::index_type index_t;

#ifndef CXX11_ENABLED
        typedef typename boost::remove_reference<typename boost::remove_pointer<BOOST_TYPEOF( (boost::fusion::at
                < index_t>(m_local_domain.m_local_args)) )>::type>::type storage_type;
        storage_type* const storage_=
#else
        auto const storage_ =
#endif
            boost::fusion::at
            < index_t>(m_local_domain.m_local_args);

        GRIDTOOLS_STATIC_ASSERT((is_accessor<Accessor>::value), "Using EVAL is only allowed for an accessor type");

#ifdef CXX11_ENABLED
        using storage_type = typename std::remove_reference<decltype(*storage_)>::type;
#endif
        typename storage_type::value_type * RESTRICT real_storage_pointer=
                static_cast<typename storage_type::value_type*>(storage_pointer);

        return *(real_storage_pointer+offset);
    }


//private:

//    template <typename GridType_>
//    struct get_pointer {
//        template <typename PlcType>
//        struct apply {
//            using type = typename GridType_::template pointer_to<typename PlcType::location_type>::type;
//        };
//    };
//    template <typename GridType_>
//    struct get_storage {
//        template <typename PlcType>
//        struct apply {
//            using type = typename GridType_::template storage_type<typename PlcType::location_type>::type;
//        };
//    };

//public:
//    using mpl_storage_types = typename boost::mpl::transform<PlcVector,
//                                                             get_storage<GridType>
//                                                             >::type;

//    using storage_types = typename boost::fusion::result_of::as_vector<mpl_storage_types>::type;

//    using mpl_pointers_t_ = typename boost::mpl::transform<PlcVector,
//                                                           get_pointer<GridType>
//                                                           >::type;

//    using pointers_t = typename boost::fusion::result_of::as_vector<mpl_pointers_t_>::type;

//    using grid_type = GridType;
//    using location_type = LocationType;
//private:
//    storage_types storages;
//    pointers_t pointers;
//    grid_type const& m_grid;

//    gridtools::array<u_int, 4> m_ll_indices;

//    template <typename PointersT, typename StoragesT>
//    struct _set_pointers
//    {
//        PointersT &m_pt;
//        StoragesT const &m_st;
//        _set_pointers(PointersT& pt, StoragesT const &st): m_pt(pt), m_st(st) {}

//        template <typename Index>
//        void operator()(Index) {
//            double * ptr = const_cast<double*>(&(*(boost::fusion::at_c<Index::value>(m_st)))(0,0,0,0));

//            boost::fusion::at_c<Index::value>(m_pt) = ptr;
//        }
//    };

//    template <typename LocationT, typename PointersT, typename StoragesT, typename GridT>
//    struct _set_pointers_to
//    {
//        PointersT &m_pt;
//        StoragesT const &m_st;
//        GridT const& m_g;
//        gridtools::array<uint_t, 4> const& _m_ll_indices;

//        _set_pointers_to(PointersT& pt,
//                         StoragesT const &st,
//                         GridT const& g,
//                         gridtools::array<uint_t, 4> const & ll_ind)
//            : m_pt(pt)
//            , m_st(st)
//            , m_g(g)
//            , _m_ll_indices(ll_ind)
//        {}

//        template <typename Index>
//        void operator()(Index) {
//            double * ptr = const_cast<double*>(&(*(boost::fusion::at_c<Index::value>(m_st)))(0,0,0,0))
//                + (boost::fusion::at_c<LocationT::value>(m_g.virtual_storages())._index(&_m_ll_indices[0]));

//            boost::fusion::at_c<Index::value>(m_pt) = ptr;
//        }
//    };

//    template <int Coordinate, typename PointersT, typename GridT>
//    struct _move_pointers
//    {
//        PointersT &m_pt;
//        GridT const &m_g;

//        _move_pointers(PointersT& m_pt, GridT const& m_g): m_pt(m_pt), m_g(m_g) {}

//        template <typename Index>
//        void operator()(Index) {
//            auto value = boost::fusion::at_c<boost::mpl::at_c<PlcVector, Index::value>::type::location_type::value>
//                (m_g.virtual_storages()).template strides(Coordinate);
//            //std::cout << "Stide<" << Index::value << "> for coordinate " << Coordinate << " = " << value << std::endl;
//            boost::fusion::at_c<Index::value>(m_pt) += value;
//        }
//    };

//    template <int Coord>
//    void _increment_pointers()
//    {
//        using indices = typename boost::mpl::range_c<int, 0, boost::fusion::result_of::size<storage_types>::type::value >;
//        boost::mpl::for_each<indices>(_move_pointers<Coord, pointers_t, grid_type>(pointers, m_grid));
//    }

//    void _reset_pointers()
//    {
//        using indices = typename boost::mpl::range_c<int, 0, boost::fusion::result_of::size<storage_types>::type::value >;
//        boost::mpl::for_each<indices>(_set_pointers<pointers_t, storage_types>(pointers, storages));
//    }



//    template <typename LocationT>
//    void _set_pointers_to_ll() {
//        using indices = typename boost::mpl::range_c<int, 0, boost::fusion::result_of::size<storage_types>::type::value >;
//        boost::mpl::for_each<indices>(_set_pointers_to<LocationT, pointers_t, storage_types, grid_type>(pointers, storages, m_grid, m_ll_indices));
//    }

//public:
//    iterate_domain(storage_types const& storages, GridType const& m_grid)
//        : storages(storages)
//        , m_grid(m_grid)
//    {
//        _reset_pointers();
//    }

//    GridType const& grid() const {return m_grid;}

//    template <int Coord>
//    void inc_ll() {++m_ll_indices[Coord]; _increment_pointers<Coord>();}

//    template <typename LocationT>
//    void set_ll_ijk(u_int i, u_int j, u_int k, u_int l) {
//        m_ll_indices = {i, j, k, l};
//        _set_pointers_to_ll<LocationT>();
//    }

//    template <typename ValueType
//              , typename LocationTypeT
//              , typename Reduction
//              , typename MapF
//              , typename ...Arg0
//              >
//    double operator()(on_neighbors_impl<ValueType, LocationTypeT, Reduction, map_function<MapF, LocationTypeT, Arg0...>> onneighbors) const {
//        auto current_position = m_ll_indices;

//        const auto neighbors = m_grid.neighbors_indices_3(current_position
//                                                          , location_type()
//                                                          , onneighbors.location() );
//#ifdef _ACCESSOR_H_DEBUG_
//        std::cout << "Entry point (on map)" << current_position << " Neighbors: " << neighbors << std::endl;
//#endif
//        double result = onneighbors.value();

//        for (int i = 0; i<neighbors.size(); ++i) {
//            result = onneighbors.reduction()( _evaluate(onneighbors.map(), neighbors[i]), result );
//        }

//        return result;
//    }

//    template <typename ValueType
//              , typename LocationTypeT
//              , typename Reduction
//              , int I
//              , typename L
//              , int R
//              >
//    double operator()(on_neighbors_impl<ValueType, LocationTypeT, Reduction, ro_accessor<I,L,radius<R>>> onneighbors) const {
//        auto current_position = m_ll_indices;

//        const auto neighbors = m_grid.neighbors_indices_3(current_position
//                                                          , location_type()
//                                                          , onneighbors.location() );
//#ifdef _ACCESSOR_H_DEBUG_
//        std::cout << "Entry point (on accessor)" << current_position << " Neighbors: " << neighbors << std::endl;
//#endif

//        double result = onneighbors.value();

//        for (int i = 0; i<neighbors.size(); ++i) {
//            result = onneighbors.reduction()( _evaluate(onneighbors.map(), neighbors[i]), result );
//        }

//        return result;
//    }

//    template <int I, typename LT>
//    double const operator()(ro_accessor<I,LT> const& arg) const {
//        //std::cout << "ADDR " << std::hex << (boost::fusion::at_c<I>(pointers)) << std::dec << std::endl;
//        return *(boost::fusion::at_c<I>(pointers));
//    }
//    template <int I, typename LT>
//    double& operator()(accessor<I,LT> const& arg) const {
//        //std::cout << "ADDR " << std::hex << (boost::fusion::at_c<I>(pointers)) << std::dec << std::endl;
//        return *(boost::fusion::at_c<I>(pointers));
//    }


//private:

    template <int I, typename L, int R, typename IndexArray>
    //TODO return the right value, instead of double
    double _evaluate(ro_accessor<I,L,radius<R>>, IndexArray const& position) const {
        typedef ro_accessor<I,L,radius<R>> accessor_t;
        int offset = m_grid_topology.ll_offset(position, typename accessor<I,L>::location_type());

        return get_raw_value(accessor_t(), (data_pointer())[current_storage<(I==0)
                , local_domain_t, typename accessor_t::type >::value], offset);
    }

//    template <typename MapF, typename LT, typename Arg0, typename IndexArray>
//    double _evaluate(map_function<MapF, LT, Arg0> map, IndexArray const& position) const {
//#ifdef _ACCESSOR_H_DEBUG_
//        std::cout << "_evaluate(map_function<MapF, LT, Arg0>...) " << LT() << ", " << position << std::endl;
//#endif
//        int offset = m_grid.ll_offset(position, map.location());
//        return map.function()(_evaluate(map.template argument<0>(), position));
//    }

//    template <typename MapF, typename LT, typename Arg0, typename Arg1, typename IndexArray>
//    double _evaluate(map_function<MapF, LT, Arg0, Arg1> map, IndexArray const& position) const {
//#ifdef _ACCESSOR_H_DEBUG_
//        std::cout << "_evaluate(map_function<MapF, LT, Arg0, Arg1>...) " << LT() << ", " << position << std::endl;
//#endif
//        int offset = m_grid.ll_offset(position, map.location());
//        return map.function()(_evaluate(map.template argument<0>(), position)
//                              , _evaluate(map.template argument<1>(), position));
//    }

//    template <typename ValueType
//              , typename LocationTypeT
//              , typename Reduction
//              , typename Map
//              , typename IndexArray>
//    double _evaluate(on_neighbors_impl<ValueType, LocationTypeT, Reduction, Map > onn, IndexArray const& position) const {
//        const auto neighbors = m_grid.neighbors_indices_3(position
//                                                          , onn.location()
//                                                          , onn.location() );
//#ifdef _ACCESSOR_H_DEBUG_
//        std::cout << "_evaluate(on_neighbors_impl<ValueType, ...) " << LocationTypeT() << ", " << position << " Neighbors: " << neighbors << std::endl;
//#endif

//        double result = onn.value();

//        for (int i = 0; i<neighbors.size(); ++i) {
//            result = onn.reduction()(_evaluate(onn.map(), neighbors[i]), result);
//        }

//        return result;
//    }

};

} //namespace gridtools
