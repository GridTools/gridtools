/*
 * GridTools Libraries
 * Copyright (c) 2019, ETH Zurich
 *
 * SPDX-License-Identifier: BSD-3-Clause
 */

#if (!defined(BOOST_PP_VARIADICS) || (BOOST_PP_VARIADICS < 1))
// defining BOOST_PP_VARIADICS 1 here might be too late, therefore we leave it to the user
#error \
    "GRIDTOOLS ERROR=> For the repository you need to \"#define BOOST_PP_VARIADICS 1\" before the first include of any boost preprocessor file.")
#endif

#include "../../c_bindings/export.hpp"
#include "../../common/boost_pp_generic_macros.hpp"
#include "../fortran_array_adapter.hpp"
#include "boost/preprocessor/seq.hpp"
#include "boost/variant.hpp"
#include "repository_macro_helpers.hpp"
#include <boost/mpl/contains.hpp>
#include <boost/mpl/vector.hpp>
#include <boost/preprocessor/control/expr_if.hpp>
#include <boost/preprocessor/list.hpp>
#include <boost/preprocessor/selection/max.hpp>
#include <boost/preprocessor/selection/min.hpp>
#include <boost/preprocessor/tuple.hpp>

#ifndef GT_PARSE_PREPROCESSOR
#include "../../common/defs.hpp"
#include <unordered_map>
#endif

#ifndef GT_REPO_GETTER_PREFIX
#define GT_REPO_GETTER_PREFIX
#endif

/*
 * @brief makes a data_store data member from the tuple (DataStoreType, MemberName).
 */
#define GT_REPO_make_members_data_stores(r, data, tuple) \
    GT_REPO_data_stores_get_typename(tuple) GT_REPO_data_stores_get_member_name_underscore(tuple);

/*
 * @brief makes a storage_info data member from the tuple (DataStoreType, DimVector). The type is
 * DataStoreType::storage_info_t, the member name is DataStoreType_info
 */
#define GT_REPO_make_members_storage_infos(r, data, tuple)                              \
    typename GT_REPO_data_store_types_get_typename(tuple)::storage_info_t BOOST_PP_CAT( \
        GT_REPO_data_store_types_get_typename(tuple), _info);

/*
 * @brief makes a getter for a data_store member variable (const and non-const version)
 */
#define GT_REPO_make_data_store_getter(r, data, tuple)                                            \
    GT_REPO_data_stores_get_typename(tuple) &                                                     \
        BOOST_PP_CAT(GT_REPO_GETTER_PREFIX, GT_REPO_data_stores_get_member_name(tuple))() {       \
        return GT_REPO_data_stores_get_member_name_underscore(tuple);                             \
    }                                                                                             \
    const GT_REPO_data_stores_get_typename(tuple) &                                               \
        BOOST_PP_CAT(GT_REPO_GETTER_PREFIX, GT_REPO_data_stores_get_member_name(tuple))() const { \
        return GT_REPO_data_stores_get_member_name_underscore(tuple);                             \
    }

/*
 * @brief makes a comma separated list of data_store types
 */
#define GT_REPO_enum_of_data_store_types_helper(r, data, n, tuple) \
    BOOST_PP_COMMA_IF(n) GT_REPO_data_store_types_get_typename(tuple)
#define GT_REPO_enum_of_data_store_types(data_store_types_seq) \
    BOOST_PP_SEQ_FOR_EACH_I(GT_REPO_enum_of_data_store_types_helper, ~, data_store_types_seq)

/*
 * @brief general helper for constructors
 */
// helper to create the initializer for data stores
#define GT_REPO_make_ctor_initializer_for_data_stores(r, data, n, tuple) \
    BOOST_PP_COMMA_IF(n)                                                 \
    GT_REPO_data_stores_get_member_name_underscore(tuple)(               \
        BOOST_PP_CAT(GT_REPO_data_stores_get_typename(tuple), _info),    \
        BOOST_PP_STRINGIZE(GT_REPO_data_stores_get_member_name(tuple)))
// put a data_store in the data_store_map_
#define GT_REPO_ctor_init_map(r, data, tuple)                                               \
    data_store_map_.emplace(BOOST_PP_STRINGIZE(GT_REPO_data_stores_get_member_name(tuple)), \
        GT_REPO_data_stores_get_member_name_underscore(tuple));

/*
 * @brief GT_REPO_make_ctor_storage_infos makes a constructor with storage-infos for each of the data_store types
 * provided in data_store_types_seq
 */
// helper to create the parameters
#define GT_REPO_make_ctor_storage_infos_params(r, data, n, data_store)                       \
    BOOST_PP_COMMA_IF(n)                                                                     \
    typename GT_REPO_data_store_types_get_typename(data_store)::storage_info_t BOOST_PP_CAT( \
        GT_REPO_data_store_types_get_typename(data_store), _info)
// helper to create the initializer for the storage_infos from storage_infos
#define GT_REPO_make_ctor_storage_infos_initializer_for_storage_infos(r, data, n, tuple) \
    BOOST_PP_COMMA_IF(n)                                                                 \
    BOOST_PP_CAT(GT_REPO_data_store_types_get_typename(tuple), _info)                    \
    (BOOST_PP_CAT(GT_REPO_data_store_types_get_typename(tuple), _info))

#define GT_REPO_make_ctor_storage_infos(name, data_store_types_seq, data_stores_seq)                   \
    name(BOOST_PP_SEQ_FOR_EACH_I(GT_REPO_make_ctor_storage_infos_params, ~, data_store_types_seq))     \
        : BOOST_PP_SEQ_FOR_EACH_I(                                                                     \
              GT_REPO_make_ctor_storage_infos_initializer_for_storage_infos, ~, data_store_types_seq), \
          BOOST_PP_SEQ_FOR_EACH_I(GT_REPO_make_ctor_initializer_for_data_stores, ~, data_stores_seq) { \
        BOOST_PP_SEQ_FOR_EACH(GT_REPO_ctor_init_map, ~, data_stores_seq)                               \
    }

/*
 * @brief GT_REPO_make_ctor_dims makes a constructor with uint_t for dimensions as described below
 * TODO we should allow a version which does allocate the storage_infos but not yet the data_stores as
 * for the interface we don't know yet if we want to allocate them for ptr sharing  mode (external ptr)
 */
// helper to generate the arguments of the form (dim0, dim1, ...) from the DimTuple (0,1,...)
#define GT_REPO_make_dim_args_helper(r, data, n, dim_value) BOOST_PP_COMMA_IF(n) BOOST_PP_CAT(dim, dim_value)
#define GT_REPO_make_dim_args(tuple) \
    (BOOST_PP_LIST_FOR_EACH_I(GT_REPO_make_dim_args_helper, ~, BOOST_PP_TUPLE_TO_LIST(tuple)))
// helper to create the initializer for the storage_infos from dimensions
#define GT_REPO_make_ctor_dims_initializer_for_storage_infos_from_dims(r, data, n, tuple) \
    BOOST_PP_COMMA_IF(n)                                                                  \
    BOOST_PP_CAT(GT_REPO_data_store_types_get_typename(tuple), _info)                     \
    GT_REPO_make_dim_args(GT_REPO_data_store_types_get_dim_tuple(tuple))
#define GT_REPO_make_ctor_dims(name, data_store_types_seq, data_stores_seq)                                   \
    name(BOOST_PP_ENUM_PARAMS(BOOST_PP_ADD(GT_REPO_max_dim(data_store_types_seq), 1), gridtools::uint_t dim)) \
        : BOOST_PP_SEQ_FOR_EACH_I(                                                                            \
              GT_REPO_make_ctor_dims_initializer_for_storage_infos_from_dims, ~, data_store_types_seq),       \
          BOOST_PP_SEQ_FOR_EACH_I(GT_REPO_make_ctor_initializer_for_data_stores, ~, data_stores_seq) {        \
        BOOST_PP_SEQ_FOR_EACH(GT_REPO_ctor_init_map, ~, data_stores_seq)                                      \
    }
/*
 * @brief makes the dims constructor only if a DimTuple is provided in the data_store_types sequence of tuples
 * (DataStoreType, DimTuple)
 */
#define GT_REPO_make_ctor_dims_if_has_dims(name, data_store_types_seq, data_stores_seq)     \
    BOOST_PP_IF(GT_REPO_has_dim(data_store_types_seq), GT_REPO_make_ctor_dims, GT_PP_EMPTY) \
    (name, data_store_types_seq, data_stores_seq)

/*
 * @brief defines "data_store_type_vector" as boost::mpl::vector of data_store types. Used to assert that all
 * data_stores are present in the sequence of data_store types
 */
#define GT_REPO_make_member_vector_of_data_store_types(data_store_types_seq) \
    using data_store_type_vector = boost::mpl::vector<GT_REPO_enum_of_data_store_types(data_store_types_seq)>

/*
 * @brief assert that all data_stores are present in the sequence of data_store types
 */
#define GT_REPO_is_valid_data_store_type(r, data, tuple)                                                      \
    GT_STATIC_ASSERT(                                                                                         \
        (boost::mpl::contains<data_store_type_vector, GT_REPO_data_stores_get_typename(tuple)>::type::value), \
        "At least one of the types in the data_store sequence is not defined in the sequence of data_store types.");

/*
 * @brief assert that all types which are passed in the sequence of data_store types are actually data_stores
 */
#define GT_REPO_is_data_store(r, data, data_store_type_tuple)                                                         \
    GT_STATIC_ASSERT((gridtools::is_data_store<GT_REPO_data_store_types_get_typename(data_store_type_tuple)>::value), \
        "At least one of the arguments passed to the repository as data_store type is not a data_store");

/*
 * @brief creates a boost::variant with implicit conversion to its types (if the compiler supports it)
 */
#if defined(__GNUC__) && (__GNUC__ >= 5) ||                                      \
    defined(__clang__) && !defined(__APPLE_CC__) &&                              \
        (__clang_major__ == 3 && __clang_minor__ >= 9 || __clang_major__ > 4) || \
    defined(__clang__) && defined(__APPLE_CC__) && __APPLE_CC__ > 8000
// this choice of host compilers implies CUDA >= 8
#define GT_REPOSITORY_HAS_VARIANT_WITH_IMPLICIT_CONVERSION
#endif

#ifdef GT_REPOSITORY_HAS_VARIANT_WITH_IMPLICIT_CONVERSION
#define GT_REPO_make_variant(name, data_store_types_seq) GT_PP_MAKE_VARIANT(name, data_store_types_seq)
#else
// use "normal" boost::variant on unsupported compilers
#define GT_REPO_make_variant(name, data_store_types_seq) \
    using name = boost::variant<GT_PP_TUPLE_ELEM_FROM_SEQ_AS_ENUM(0, data_store_types_seq)>;
#endif
#define GT_REPO_get_binding_name(fortran_name, member_name) \
    BOOST_PP_CAT(BOOST_PP_CAT(BOOST_PP_CAT(set_, fortran_name), _), member_name)
#define GT_REPO_make_binding_helper(repo_name, fortran_name, member_name, type)      \
    void BOOST_PP_CAT(GT_REPO_get_binding_name(fortran_name, member_name), _impl)(   \
        repo_name & repo, gridtools::fortran_array_adapter<type> view) {             \
        transform(repo.BOOST_PP_CAT(GT_REPO_GETTER_PREFIX, member_name)(), view);    \
    }                                                                                \
    GT_EXPORT_BINDING_WRAPPED_2(GT_REPO_get_binding_name(fortran_name, member_name), \
        BOOST_PP_CAT(GT_REPO_get_binding_name(fortran_name, member_name), _impl));

#define GT_REPO_make_binding(r, data, tuple)                     \
    GT_REPO_make_binding_helper(BOOST_PP_TUPLE_ELEM(2, 0, data), \
        BOOST_PP_TUPLE_ELEM(2, 1, data),                         \
        GT_REPO_data_stores_get_member_name(tuple),              \
        GT_REPO_data_store_types_get_typename(tuple))

/*
 * @brief main macro to generate a repository
 * @see GT_MAKE_REPOSITORY
 */
#define GT_MAKE_REPOSITORY_helper(name, data_store_types_seq, data_stores_seq)                                        \
    class name {                                                                                                      \
      private:                                                                                                        \
        GT_REPO_make_variant(                                                                                         \
            BOOST_PP_CAT(name, _variant), data_store_types_seq) /* in class definition of a special boost::variant    \
                                                                   class with automatic type conversion*/             \
            BOOST_PP_SEQ_FOR_EACH(                                                                                    \
                GT_REPO_make_members_storage_infos, ~, data_store_types_seq) /* generate storage_info members*/       \
            BOOST_PP_SEQ_FOR_EACH(                                                                                    \
                GT_REPO_make_members_data_stores, ~, data_stores_seq) /* generate data_store members*/                \
            std::unordered_map<std::string, BOOST_PP_CAT(name, _variant)> data_store_map_; /* map for data_stores */  \
        BOOST_PP_SEQ_FOR_EACH(GT_REPO_is_data_store,                                                                  \
            ~,                                                                                                        \
            data_store_types_seq) /*assert that all types passed in data_store_types are actually data_stores*/       \
        GT_REPO_make_member_vector_of_data_store_types(data_store_types_seq); /* mpl vector of data_store types */    \
        BOOST_PP_SEQ_FOR_EACH(GT_REPO_is_valid_data_store_type,                                                       \
            ~,                                                                                                        \
            data_stores_seq) /*assert that the type of each data_store is provided in the data_store_types seq*/      \
      public:                                                                                                         \
        GT_REPO_make_ctor_storage_infos(                                                                              \
            name, data_store_types_seq, data_stores_seq); /* generate ctor with storage_infos */                      \
        GT_REPO_make_ctor_dims_if_has_dims(                                                                           \
            name, data_store_types_seq, data_stores_seq)                          /* generate ctor with dimensions */ \
            name(const name &) = delete;                                          /* non-copyable */                  \
        name(name &&) = default;                                                  /* movable */                       \
        BOOST_PP_SEQ_FOR_EACH(GT_REPO_make_data_store_getter, ~, data_stores_seq) /* getter for each data_store */    \
        auto data_stores() -> decltype(data_store_map_) & { return data_store_map_; } /* getter for data_store map */ \
        const std::unordered_map<std::string, BOOST_PP_CAT(name, _variant)> &data_stores() const {                    \
            return data_store_map_;                                                                                   \
        } /* const getter for data_store map */                                                                       \
    };

/*
 * @brief main macro to generate the fortran bindings for a repository
 * @see GT_MAKE_REPOSITORY
 */
#define GT_MAKE_REPOSITORY_BINDINGS_helper(name, fortran_name, data_stores_seq) \
    BOOST_PP_SEQ_FOR_EACH(GT_REPO_make_binding, (name, fortran_name), data_stores_seq)

/*
 * @brief entry for the user
 * @param name class name for the repository
 * @param data_store_types_seq BOOST_PP sequence of tuples of the form (DataStoreType, DimensionTuple)
 * @param data_stores_seq BOOST_PP sequence of tuples of the form (DataStoreType, VariableName)
 *
 * The DimensionTuple is optional.
 * Example with Dimensions:
 * (IJKDataStore,(0,1,2))(IKDataStore(0,2)) will generate a repository with two constructors
 * a) repository( IJKDataStore::storage_info_t, IJDataStore::storage_info_t )
 *    to initialize all data_stores with their respective storage_infos
 * b) repository( uint_t dim0, uint_t dim1, uint_t dim2 )
 *    will initialize storage_infos:
 *      IJKDataStore::storage_info_t(dim0,dim1,dim2) and
 *      IKDataStore::storage_info_t(dim0,dim2)
 *
 * Main macro is GT_MAKE_REPOSITORY_helper. Here we just add extra parenthesis to the input to make user-code
 * look nicer (no double parenthesis)
 */
#define GT_MAKE_REPOSITORY(name, data_store_types_seq, data_stores_seq) \
    GT_MAKE_REPOSITORY_helper(                                          \
        name, BOOST_PP_VARIADIC_SEQ_TO_SEQ(data_store_types_seq), BOOST_PP_VARIADIC_SEQ_TO_SEQ(data_stores_seq))

/*
 * @brief Creates the fortran bindings for the repository. Must be called from a cpp file.
 * @param name class name for the repository
 * @param fortran_name name that will be used to identify the repository in the fortran binding
 * @param data_stores_seq BOOST_PP sequence of tuples of the form (DataStoreType, VariableName)
 *
 * Main macro is GT_MAKE_REPOSITORY_BINDINGS_helper. Here we just add extra parenthesis to the input to make
 * user-code look nicer (no double parenthesis)
 *
 * Suppose you have a repository with name = "CRep", fortran_name = "FRep" and datastores named "u" and "v". This will
 * generate the following fortran bindings:
 *     set_FRep_u(repo, arr) sets CRep.u()
 *     set_FRep_v(repo, arr) sets CRep.v()
 */
#define GT_MAKE_REPOSITORY_BINDINGS(name, fortran_name, data_stores_seq) \
    GT_MAKE_REPOSITORY_BINDINGS_helper(name, fortran_name, BOOST_PP_VARIADIC_SEQ_TO_SEQ(data_stores_seq))
