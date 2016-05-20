#pragma once
#include "location_type.hpp"
#include <type_traits>
#include <boost/type_traits/remove_reference.hpp>
#include "common/generic_metafunctions/apply_to_sequence.hpp"
#include "common/generic_metafunctions/vector_to_set.hpp"
#include "common/generic_metafunctions/variadic_to_vector.hpp"
#include "common/generic_metafunctions/variadic_typedef.hpp"
#include "common/array.hpp"
#include "../../common/explode_array.hpp"
#include "common/generic_metafunctions/remove_restrict_reference.hpp"
#include "stencil-composition/iterate_domain_impl_metafunctions.hpp"
#include "stencil-composition/total_storages.hpp"
#include "stencil-composition/iterate_domain_aux.hpp"
#include "stencil-composition/icosahedral_grids/accessor_metafunctions.hpp"
#include "on_neighbors.hpp"
#include "../iterate_domain_fwd.hpp"

namespace gridtools {

    // TODO move this to the appropiate file
    template < typename EsfSequence >
    struct extract_location_type {
        GRIDTOOLS_STATIC_ASSERT((is_sequence_of< EsfSequence, is_esf_descriptor >::value), "Error: wrong type");
        typedef typename apply_to_sequence< EsfSequence, esf_get_location_type >::type location_type_seq_t;
        typedef typename vector_to_set< location_type_seq_t >::type location_type_set_t;

        GRIDTOOLS_STATIC_ASSERT((boost::mpl::size< location_type_set_t >::value == 1),
            "Error: multiple ESFs were used with different location types."
            " Currently all esf must be specified on the same location type. "
            "Future releases will relax this restriction");
        typedef typename boost::mpl::front< location_type_set_t >::type type;
    };

    /**
       This class is basically the iterate domain. It contains the
       ways to access data and the implementation of iterating on neighbors.
     */
    // template <typename PlcVector, typename GridType, typename LocationType>
    template < typename IterateDomainImpl >
    struct iterate_domain {
        typedef iterate_domain< IterateDomainImpl > type;

        typedef typename iterate_domain_impl_arguments< IterateDomainImpl >::type iterate_domain_arguments_t;
        typedef typename iterate_domain_arguments_t::local_domain_t local_domain_t;

        typedef typename iterate_domain_arguments_t::processing_elements_block_size_t processing_elements_block_size_t;

        typedef typename iterate_domain_arguments_t::backend_ids_t backend_ids_t;
        typedef typename iterate_domain_arguments_t::grid_t::grid_topology_t grid_topology_t;
        typedef typename iterate_domain_arguments_t::esf_sequence_t esf_sequence_t;
        typedef typename extract_location_type< esf_sequence_t >::type location_type_t;

        typedef typename local_domain_t::esf_args esf_args_t;

        typedef typename backend_traits_from_id< backend_ids_t::s_backend_id >::template select_iterate_domain_cache<
            iterate_domain_arguments_t >::type iterate_domain_cache_t;

        typedef typename iterate_domain_cache_t::ij_caches_map_t ij_caches_map_t;

        GRIDTOOLS_STATIC_ASSERT((is_local_domain< local_domain_t >::value), "Internal Error: wrong type");
        typedef typename boost::remove_pointer<
            typename boost::mpl::at_c< typename local_domain_t::mpl_storages, 0 >::type >::type::value_type value_type;

        typedef typename local_domain_t::storage_metadata_map metadata_map_t;
        typedef typename local_domain_t::actual_args_type actual_args_type;

        // the number of different storage metadatas used in the current functor
        static const uint_t N_META_STORAGES = boost::mpl::size< metadata_map_t >::value;
        // the number of storages  used in the current functor
        static const uint_t N_STORAGES = boost::mpl::size< actual_args_type >::value;
        // the total number of snapshot (one or several per storage)
        static const uint_t N_DATA_POINTERS = total_storages< actual_args_type,
            boost::mpl::size< typename local_domain_t::mpl_storages >::type::value >::value;

        typedef array< void * RESTRICT, N_DATA_POINTERS > data_pointer_array_t;
        typedef strides_cached< N_META_STORAGES - 1, typename local_domain_t::storage_metadata_vector_t >
            strides_cached_t;

        /**@brief local class instead of using the inline (cond)?a:b syntax, because in the latter both branches get
         * compiled (generating sometimes a compile-time overflow) */
        template < bool condition, typename LocalD, typename Accessor >
        struct current_storage;

        template < typename LocalD, typename Accessor >
        struct current_storage< true, LocalD, Accessor > {
            static const uint_t value = 0;
        };

        template < typename LocalD, typename Accessor >
        struct current_storage< false, LocalD, Accessor > {
            static const uint_t value =
                (total_storages< typename LocalD::local_args_type, Accessor::index_type::value >::value);
        };

        /**
         * metafunction that computes the return type of all operator() of an accessor
         */
        template < typename Accessor >
        struct accessor_return_type {
            typedef typename ::gridtools::accessor_return_type< Accessor, iterate_domain_arguments_t >::type type;
        };

        template < typename T >
        struct map_return_type;

        template < typename MapF, typename LT, typename Arg0, typename... Args >
        struct map_return_type< map_function< MapF, LT, Arg0, Args... > > {
            GRIDTOOLS_STATIC_ASSERT((is_accessor< Arg0 >::value), "Error");
            typedef typename remove_restrict_reference< typename accessor_return_type< Arg0 >::type >::type type;
        };

        typedef typename compute_readonly_args_indices< typename iterate_domain_arguments_t::esf_sequence_t >::type
            readonly_args_indices_t;

      private:
        GRIDTOOLS_STATIC_ASSERT((N_META_STORAGES <= grid_topology_t::n_locations::value),
            "We can not have more meta storages"
            "than location types. Data fields for other grids are not yet supported");
        local_domain_t const &m_local_domain;
        grid_topology_t const &m_grid_topology;
        typedef array< int_t, N_META_STORAGES > array_index_t;
        // TODOMEETING do we need m_index?
        array_index_t m_index;

        array< uint_t, 4 > m_grid_position;

      public:
        /**@brief constructor of the iterate_domain struct

           It assigns the storage pointers to the first elements of
           the data fields (for all the data_fields present in the
           current evaluation), and the indexes to access the data
           fields (one index per storage instance, so that one index
           might be shared among several data fileds)
        */
        GT_FUNCTION
        iterate_domain(local_domain_t const &local_domain_, grid_topology_t const &grid_topology)
            : m_local_domain(local_domain_), m_grid_topology(grid_topology) {}

        /**
           @brief returns the array of pointers to the raw data
        */
        GT_FUNCTION
        data_pointer_array_t const &RESTRICT data_pointer() const {
            return static_cast< IterateDomainImpl const * >(this)->data_pointer_impl();
        }

        /**
           @brief returns the array of pointers to the raw data
        */
        GT_FUNCTION
        data_pointer_array_t &RESTRICT data_pointer() {
            return static_cast< IterateDomainImpl * >(this)->data_pointer_impl();
        }

        /**
           @brief returns the strides as const reference
        */
        GT_FUNCTION
        strides_cached_t const &RESTRICT strides() const {
            return static_cast< IterateDomainImpl const * >(this)->strides_impl();
        }

        /**
           @brief returns the strides as const reference
        */
        GT_FUNCTION
        strides_cached_t &RESTRICT strides() { return static_cast< IterateDomainImpl * >(this)->strides_impl(); }

        /** This functon set the addresses of the data values  before the computation
            begins.

            The EU stands for ExecutionUnit (thich may be a thread or a group of
            threasd. There are potentially two ids, one over i and one over j, since
            our execution model is parallel on (i,j). Defaulted to 1.
        */
        template < typename BackendType >
        GT_FUNCTION void assign_storage_pointers() {
            const uint_t EU_id_i = BackendType::processing_element_i();
            const uint_t EU_id_j = BackendType::processing_element_j();
            boost::mpl::for_each< typename reversed_range< uint_t, 0, N_STORAGES >::type >(
                assign_storage_functor< BackendType,
                    data_pointer_array_t,
                    typename local_domain_t::local_args_type,
                    typename local_domain_t::local_metadata_type,
                    metadata_map_t,
                    processing_elements_block_size_t >(
                    data_pointer(), m_local_domain.m_local_args, m_local_domain.m_local_metadata, EU_id_i, EU_id_j));
        }

        /**
           @brief recursively assignes all the strides

           copies them from the
           local_domain.m_local_metadata vector, and stores them into an instance of the
           \ref strides_cached class.
         */
        template < typename BackendType, typename Strides >
        GT_FUNCTION void assign_stride_pointers() {
            GRIDTOOLS_STATIC_ASSERT((is_strides_cached< Strides >::value), "internal error type");
            boost::mpl::for_each< metadata_map_t >(assign_strides_functor< BackendType,
                Strides,
                typename boost::fusion::result_of::as_vector< typename local_domain_t::local_metadata_type >::type,
                processing_elements_block_size_t >(strides(), m_local_domain.m_local_metadata));
        }

        /**@brief method for initializing the index */
        template < ushort_t Coordinate >
        GT_FUNCTION void initialize(uint_t const initial_pos = 0, uint_t const block = 0) {
            boost::mpl::for_each< metadata_map_t >(initialize_index_functor< Coordinate,
                strides_cached_t,
                typename boost::fusion::result_of::as_vector< typename local_domain_t::local_metadata_type >::type,
                array_index_t >(
                strides(), boost::fusion::as_vector(m_local_domain.m_local_metadata), initial_pos, block, m_index));
            static_cast< IterateDomainImpl * >(this)->template initialize_impl< Coordinate >();

            m_grid_position[Coordinate] = initial_pos;
        }

        /**@brief method for incrementing by 1 the index when moving forward along the given direction
           \tparam Coordinate dimension being incremented
           \tparam Execution the policy for the increment (e.g. forward/backward)
         */
        template < ushort_t Coordinate, typename Steps >
        GT_FUNCTION void increment() {
            boost::mpl::for_each< metadata_map_t >(increment_index_functor< Coordinate,
                strides_cached_t,
                typename boost::fusion::result_of::as_vector< typename local_domain_t::local_metadata_type >::type,
                array_index_t >(
                boost::fusion::as_vector(m_local_domain.m_local_metadata), Steps::value, m_index, strides()));
            static_cast< IterateDomainImpl * >(this)->template increment_impl< Coordinate, Steps >();
            m_grid_position[Coordinate] += Steps::value;
        }

        /**@brief method for incrementing the index when moving forward along the given direction

           \param steps_ the increment
           \tparam Coordinate dimension being incremented
         */
        template < ushort_t Coordinate >
        GT_FUNCTION void increment(int_t steps_) {
            boost::mpl::for_each< metadata_map_t >(increment_index_functor< Coordinate,
                strides_cached_t,
                typename boost::fusion::result_of::as_vector< typename local_domain_t::local_metadata_type >::type,
                array_index_t >(boost::fusion::as_vector(m_local_domain.m_local_metadata), steps_, m_index, strides()));
            static_cast< IterateDomainImpl * >(this)->template increment_impl< Coordinate >(steps_);

            m_grid_position[Coordinate] += steps_;
        }

        /**@brief getter for the index array */
        // TODO simplify this using just loops
        GT_FUNCTION
        void get_index(array< int_t, N_META_STORAGES > &index) const {
            for (int_t i = 0; i < N_META_STORAGES; ++i) {
                index[i] = m_index[i];
            }
        }

        GT_FUNCTION
        array< uint_t, 4 > const &position() const { return m_grid_position; }

        /**@brief getter for the index array */
        GT_FUNCTION
        void get_position(array< uint_t, 4 > &position) const { position = m_grid_position; }

        /**@brief method for setting the index array */
        template < typename Value >
        GT_FUNCTION void set_index(array< Value, N_META_STORAGES > const &index) {
            for (int_t i = 0; i < N_META_STORAGES; ++i) {
                m_index[i] = index[i];
            }
        }

        GT_FUNCTION
        void set_index(int index) {
            for (int_t i = 0; i < N_META_STORAGES; ++i) {
                m_index[i] = index;
            }
        }

        GT_FUNCTION
        void set_position(array< uint_t, 4 > const &position) { m_grid_position = position; }

        template < uint_t ID,
            enumtype::intend intend,
            typename LocationType,
            typename Extent,
            ushort_t FieldDimensions >
        GT_FUNCTION typename accessor_return_type< accessor< ID, intend, LocationType, Extent, FieldDimensions > >::type
        operator()(accessor< ID, intend, LocationType, Extent, FieldDimensions > const &accessor_) const {
            typedef accessor< ID, intend, LocationType, Extent, FieldDimensions > accessor_t;
            return get_value(accessor_,
                (data_pointer())[current_storage< (ID == 0), local_domain_t, typename accessor_t::type >::value]);
        }

        template < typename ValueType, typename LocationTypeT, typename Reduction, typename MapF, typename... Arg0 >
        GT_FUNCTION ValueType operator()(
            on_neighbors_impl< ValueType, LocationTypeT, Reduction, map_function< MapF, LocationTypeT, Arg0... > >
                onneighbors) const {
            auto current_position = m_grid_position;

            const auto neighbors =
                m_grid_topology.neighbors_indices_3(current_position, location_type_t(), onneighbors.location());
            ValueType result = onneighbors.value();

            for (int i = 0; i < neighbors.size(); ++i) {
                result = onneighbors.reduction()(_evaluate(onneighbors.template map< 0 >(), neighbors[i]), result);
            }

            return result;
        }

        /**
         * helper to dereference the value (using an iterate domain) of an accessor
         * (specified with an Index from within a variadic pack of Accessors). It is meant to be used as
         * a functor of a apply_gt_integer_sequence, where the Index is provided from the integer sequence
         * @tparam ValueType value type of the computation
         */
        template < typename ValueType >
        struct it_domain_evaluator {

            /**
             * @tparam Idx index being processed from within an apply_gt_integer_sequence
             */
            template < int Idx >
            struct apply_t {

                GT_FUNCTION
                constexpr apply_t() {}

                /**
                 * @tparam Neighbors type locates the position of a neighbor element in the grid. If can be:
                 *     * a quad of values indicating the {i,c,j,k} positions or
                 *     * an integer indicating the absolute index in the storage
                 * @tparam IterateDomain is an iterate domain
                 * @tparam Accessors variadic pack of accessors being processed by the apply_gt_integer_sequence
                 *     and to be evaluated by the iterate domain
                 */
                template < typename Neighbors, typename IterateDomain, typename... Accessors >
                GT_FUNCTION static ValueType apply(
                    Neighbors const &neighbors, IterateDomain const &iterate_domain, Accessors... args_) {

                    return iterate_domain._evaluate(get_from_variadic_pack< Idx >::apply(args_...), neighbors);
                }
            };
        };

        /**
         * data structure that holds data needed by the reduce_tuple functor
         * @tparam ValueType value type of the computation
         * @tparam NeighborsArray type locates the position of a neighbor element in the grid. If can be:
         *     * a quad of values indicating the {i,c,j,k} positions or
         *     * an integer indicating the absolute index in the storage
         * @tparam Reduction this is the user lambda specified to expand the on_XXX keyword
         * @tparam IterateDomain is an iterate domain
         */
        template < typename ValueType, typename NeighborsArray, typename Reduction, typename IterateDomain >
        struct reduce_tuple_data_holder {
            Reduction const &m_reduction;
            NeighborsArray const &m_neighbors;
            IterateDomain const &m_iterate_domain;
            ValueType &m_result;

          public:
            GT_FUNCTION
            reduce_tuple_data_holder(Reduction const &reduction,
                NeighborsArray const &neighbors,
                ValueType &result,
                IterateDomain const &iterate_domain)
                : m_reduction(reduction), m_neighbors(neighbors), m_result(result), m_iterate_domain(iterate_domain) {}
        };

        /**
         * functor used to expand all the accessors arguments stored in a tuple of a on_neighbors structured.
         * The functor will process all the accessors (i.e. dereference their values of the storages given an neighbors
         * offset)
         * and call the user lambda
         * @tparam ValueType value type of the computation
         * @tparam NeighborsArray type locates the position of a neighbor element in the grid. If can be:
         *     * a quad of values indicating the {i,c,j,k} positions or
         *     * an integer indicating the absolute index in the storage
         * @tparam Reduction this is the user lambda specified to expand the on_XXX keyword
         * @tparam IterateDomain is an iterate domain
         */
        template < typename ValueType, typename NeighborsArray, typename Reduction, typename IterateDomain >
        struct reduce_tuple {

            GRIDTOOLS_STATIC_ASSERT(
                (boost::is_same<
                     typename boost::remove_const< typename boost::remove_reference< NeighborsArray >::type >::type,
                     unsigned int >::value ||
                    is_array< typename boost::remove_const<
                        typename boost::remove_reference< NeighborsArray >::type >::type >::value),
                "Error");

            GRIDTOOLS_STATIC_ASSERT((is_iterate_domain< IterateDomain >::value), "Error");

            typedef reduce_tuple_data_holder< ValueType, NeighborsArray, Reduction, IterateDomain >
                reduce_tuple_holder_t;

            template < typename... Accessors >
            GT_FUNCTION static void apply(reduce_tuple_holder_t &reducer, Accessors... args) {
                // we need to call the user functor (Reduction(arg1, arg2, ..., result) )
                // However we can make here a direct call, since we first need to dereference the address of each
                // Accessor
                // given the array with position of the neighbor being accessed (reducer.m_neighbors)
                // We make use of the apply_gt_integer_sequence in order to operate on each element of the variadic
                // pack,
                // dereference its address (it_domain_evaluator) and gather back all the arguments while calling the
                // user lambda
                // (Reduction)
                using seq =
                    apply_gt_integer_sequence< typename make_gt_integer_sequence< int, sizeof...(Accessors) >::type >;

                reducer.m_result = seq::template apply_lambda< ValueType,
                    Reduction,
                    it_domain_evaluator< ValueType >::template apply_t >(
                    reducer.m_reduction, reducer.m_result, reducer.m_neighbors, reducer.m_iterate_domain, args...);
            }
        };

        // specialization of the () operator for on_neighbors operating on accessors
        template < typename ValueType, typename LocationTypeT, typename Reduction, typename... Accessors >
        GT_FUNCTION typename boost::enable_if<
            typename is_sequence_of< typename variadic_to_vector< Accessors... >::type, is_accessor >::type,
            ValueType >::type
        operator()(on_neighbors_impl< ValueType, LocationTypeT, Reduction, Accessors... > onneighbors) const {
            auto current_position = m_grid_position;

            // the type of neighbors variable depends on the source and destination location types.
            // If iteration space (location_type_t) and destionation location type of the on_neighbors (LocationTypeT)
            // match,
            // the neighbors are described as an array of {i,c,j,k} positions, i.e. an array< array<uint_t, 4>,
            // NumNeighbors>
            // Otherwise it is a array of absolute indices in the storage, i.e. an array<uint?t, NumNeighbors>
            // The following code that reduces the values work independently of the type of neighbors returned
            const auto neighbors =
                m_grid_topology.neighbors_indices_3(current_position, location_type_t(), onneighbors.location());

            ValueType &result = onneighbors.value();

            for (int_t i = 0; i < neighbors.size(); ++i) {

                typedef decltype(neighbors[i]) neighbors_array_t;
                reduce_tuple_data_holder< ValueType, neighbors_array_t, Reduction, type > red(
                    onneighbors.reduction(), neighbors[i], result, *this);
                // since the on_neighbors store a tuple of accessors (in maps() ), we should explode the tuple,
                // so that each element of the tuple is passed as an argument of the user lambda
                // (which happens in the reduce_tuple).
                explode< void, reduce_tuple< ValueType, neighbors_array_t, Reduction, type > >(onneighbors.maps(), red);
            }

            return result;
        }

        /**@brief returns the value of the memory at the given address, plus the offset specified by the arg
           placeholder
           \param arg placeholder containing the storage ID and the offsets
           \param storage_pointer pointer to the first element of the specific data field used
        */
        // TODO This should be merged with structured grids
        template < typename Accessor, typename StoragePointer >
        GT_FUNCTION typename accessor_return_type< Accessor >::type get_value(
            Accessor const &accessor, StoragePointer &RESTRICT storage_pointer) const {

            // getting information about the storage
            typedef typename Accessor::index_type index_t;

            auto const storage_ = boost::fusion::at< index_t >(m_local_domain.m_local_args);

            GRIDTOOLS_STATIC_ASSERT(
                (is_accessor< Accessor >::value), "Using EVAL is only allowed for an accessor type");

            using storage_type = typename std::remove_reference< decltype(*storage_) >::type;
            typename storage_type::value_type *RESTRICT real_storage_pointer =
                static_cast< typename storage_type::value_type * >(storage_pointer);

            // getting information about the metadata
            typedef typename boost::mpl::at< metadata_map_t, typename storage_type::storage_info_type >::type
                metadata_index_t;

            pointer< const typename storage_type::storage_info_type > const metadata_ =
                boost::fusion::at< metadata_index_t >(m_local_domain.m_local_metadata);
            // getting the value

            // the following assert fails when an out of bound access is observed, i.e. either one of
            // i+offset_i or j+offset_j or k+offset_k is too large.
            // Most probably this is due to you specifying a positive offset which is larger than expected,
            // or maybe you did a mistake when specifying the extents in the placehoders definition
            assert(metadata_->size() > metadata_->index(m_grid_position));

            // the following assert fails when an out of bound access is observed,
            // i.e. when some offset is negative and either one of
            // i+offset_i or j+offset_j or k+offset_k is too small.
            // Most probably this is due to you specifying a negative offset which is
            // smaller than expected, or maybe you did a mistake when specifying the extents
            // in the placehoders definition.
            // If you are running a parallel simulation another common reason for this to happen is
            // the definition of an halo region which is too small in one direction
            // std::cout<<"Storage Index: "<<Accessor::index_type::value<<" + "<<(boost::fusion::at<typename
            // Accessor::index_type>(local_domain.local_args))->_index(arg.template
            // n<Accessor::n_dim>())<<std::endl;
            assert((int_t)(metadata_->index(m_grid_position)) >= 0);

            const int_t pointer_offset =
                metadata_->index(m_grid_position) +
                metadata_->_index(strides().template get< metadata_index_t::value >(), accessor);

            return *(real_storage_pointer + pointer_offset);
        }

        template < typename Accessor, typename StoragePointer >
        GT_FUNCTION typename accessor_return_type< Accessor >::type get_raw_value(
            Accessor const &accessor, StoragePointer &RESTRICT storage_pointer, const uint_t offset) const {

            // getting information about the storage
            typedef typename Accessor::index_type index_t;

            auto const storage_ = boost::fusion::at< index_t >(m_local_domain.m_local_args);

            GRIDTOOLS_STATIC_ASSERT(
                (is_accessor< Accessor >::value), "Using EVAL is only allowed for an accessor type");

            using storage_type = typename std::remove_reference< decltype(*storage_) >::type;
            typename storage_type::value_type *RESTRICT real_storage_pointer =
                static_cast< typename storage_type::value_type * >(storage_pointer);

            return *(real_storage_pointer + offset);
        }

        /**
         * It dereferences the value of an accessor given its 4d (i,c,j,k) position
         */
        template < uint_t ID,
            enumtype::intend Intend,
            typename LocationType,
            typename Extent,
            ushort_t FieldDimensions >
        GT_FUNCTION typename std::remove_reference<
            typename accessor_return_type< accessor< ID, Intend, LocationType, Extent, FieldDimensions > >::type >::type
            _evaluate(accessor< ID, Intend, LocationType, Extent, FieldDimensions >,
                array< uint_t, 4 > const &position) const {
            using accessor_t = accessor< ID, Intend, LocationType, Extent, FieldDimensions >;
            using location_type_t = typename accessor_t::location_type;
            int offset = m_grid_topology.ll_offset(position, location_type_t());

            return get_raw_value(accessor_t(),
                (data_pointer())[current_storage< (accessor_t::index_type::value == 0),
                    local_domain_t,
                    typename accessor_t::type >::value],
                offset);
        }

        /**
         * It dereferences the value of an accessor given its absolute offset
         */
        template < uint_t ID,
            enumtype::intend Intend,
            typename LocationType,
            typename Extent,
            ushort_t FieldDimensions >
        GT_FUNCTION typename std::remove_reference<
            typename accessor_return_type< accessor< ID, Intend, LocationType, Extent, FieldDimensions > >::type >::type
        _evaluate(accessor< ID, Intend, LocationType, Extent, FieldDimensions >, const uint_t offset) const {
            using accessor_t = accessor< ID, Intend, LocationType, Extent, FieldDimensions >;
            using location_type_t = typename accessor_t::location_type;

            return get_raw_value(accessor_t(),
                (data_pointer())[current_storage< (accessor_t::index_type::value == 0),
                    local_domain_t,
                    typename accessor_t::type >::value],
                offset);
        }

        template < typename MapF, typename LT, typename Arg0, typename IndexArray >
        GT_FUNCTION typename map_return_type< map_function< MapF, LT, Arg0 > >::type _evaluate(
            map_function< MapF, LT, Arg0 > const &map, IndexArray const &position) const {
            int offset = m_grid_topology.ll_offset(position, map.location());
            return map.function()(_evaluate(map.template argument< 0 >(), position));
        }

        template < typename MapF, typename LT, typename Arg0, typename Arg1, typename IndexArray >
        GT_FUNCTION typename map_return_type< map_function< MapF, LT, Arg0, Arg1 > >::type _evaluate(
            map_function< MapF, LT, Arg0, Arg1 > const &map, IndexArray const &position) const {
            int offset = m_grid_topology.ll_offset(position, map.location());
            _evaluate(map.template argument< 1 >(), position);

            return map.function()(
                _evaluate(map.template argument< 0 >(), position), _evaluate(map.template argument< 1 >(), position));
        }

        template < typename ValueType, typename LocationTypeT, typename Reduction, typename Map, typename IndexArray >
        GT_FUNCTION ValueType _evaluate(
            on_neighbors_impl< ValueType, LocationTypeT, Reduction, Map > onn, IndexArray const &position) const {

            // TODO THIS IS WRONG HERE HARDCODED EDGES
            using tt = typename grid_topology_t::edges;
            const auto neighbors = m_grid_topology.neighbors_indices_3(position, tt(), onn.location());
            ValueType result = onn.value();

            for (int i = 0; i < neighbors.size(); ++i) {
                result = onn.reduction()(_evaluate(onn.template map< 0 >(), neighbors[i]), result);
            }

            return result;
        }
    };

} // namespace gridtools
