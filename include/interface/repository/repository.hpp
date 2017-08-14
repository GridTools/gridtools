/*
  GridTools Libraries

  Copyright (c) 2017, ETH Zurich and MeteoSwiss
  All rights reserved.

  Redistribution and use in source and binary forms, with or without
  modification, are permitted provided that the following conditions are
  met:

  1. Redistributions of source code must retain the above copyright
  notice, this list of conditions and the following disclaimer.

  2. Redistributions in binary form must reproduce the above copyright
  notice, this list of conditions and the following disclaimer in the
  documentation and/or other materials provided with the distribution.

  3. Neither the name of the copyright holder nor the names of its
  contributors may be used to endorse or promote products derived from
  this software without specific prior written permission.

  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
  A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
  HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
  SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
  LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
  DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
  THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
  (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
  OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

  For information: http://eth-cscs.github.io/gridtools/
*/

#if (!defined(BOOST_PP_VARIADICS) || (BOOST_PP_VARIADICS < 1))
// defining BOOST_PP_VARIADICS 1 here might be too late, therefore we leave it to the user
#error \
    "GRIDTOOLS ERROR=> For the repository you need to \"#define BOOST_PP_VARIADICS 1\" before the first include of any boost preprocessor file.")
#endif

#include <unordered_map>
#include "boost/variant.hpp"
#include "boost/preprocessor/seq.hpp"
#include <boost/preprocessor/tuple.hpp>
#include <boost/preprocessor/list.hpp>
#include <boost/preprocessor/selection/max.hpp>
#include <boost/preprocessor/selection/min.hpp>
#include <boost/preprocessor/control/expr_if.hpp>
#include <boost/preprocessor/facilities/identity.hpp>
#include <boost/mpl/vector.hpp>
#include <boost/mpl/contains.hpp>
#include "../../common/defs.hpp"

// helper: adds double parenthesis to make user-code sequences look nicer
#define PP_DETAIL_SEQ_DOUBLE_PARENS_0(...) ((__VA_ARGS__)) PP_DETAIL_SEQ_DOUBLE_PARENS_1
#define PP_DETAIL_SEQ_DOUBLE_PARENS_1(...) ((__VA_ARGS__)) PP_DETAIL_SEQ_DOUBLE_PARENS_0
#define PP_DETAIL_SEQ_DOUBLE_PARENS_0_END
#define PP_DETAIL_SEQ_DOUBLE_PARENS_1_END
#define PP_SEQ_DOUBLE_PARENS(seq) BOOST_PP_CAT(PP_DETAIL_SEQ_DOUBLE_PARENS_0 seq, _END)

// getter for fieldtype_tuple
// fieldtype_tuple = (name, tuple_of_dimensions)
#define GTREPO_fieldtype_name(fieldtype_tuple) BOOST_PP_TUPLE_ELEM(0, fieldtype_tuple)
#define GTREPO_fieldtype_dims_tuple(fieldtype_tuple) BOOST_PP_TUPLE_ELEM(1, fieldtype_tuple)
// getter for field_tuple
// field_tuple = (type, name)
#define GTREPO_field_type(tuple) BOOST_PP_TUPLE_ELEM(0, tuple)
#define GTREPO_field_name(tuple) BOOST_PP_TUPLE_ELEM(1, tuple)

#define GTREPO_name(name) BOOST_PP_CAT(name, _)
#define GTREPO_make_members(r, data, tuple) BOOST_PP_TUPLE_ELEM(0, tuple) GTREPO_name(BOOST_PP_TUPLE_ELEM(1, tuple));
#define GTREPO_make_storage_info_members(r, data, tuple) \
    typename GTREPO_fieldtype_name(tuple)::storage_info_t BOOST_PP_CAT(GTREPO_fieldtype_name(tuple), _info);
#define GTREPO_as_string_helper(name) #name
#define GTREPO_as_string(name) GTREPO_as_string_helper(name)

// helper for ctor with storage_infos
#define GTREPO_make_ctor_args(r, data, n, data_store) \
    BOOST_PP_COMMA_IF(n)                              \
    typename GTREPO_fieldtype_name(data_store)::storage_info_t BOOST_PP_CAT(GTREPO_fieldtype_name(data_store), _info)
#define GTREPO_make_ctor_init(r, data, n, tuple) \
    BOOST_PP_COMMA_IF(n)                         \
    GTREPO_name(BOOST_PP_TUPLE_ELEM(1, tuple))(  \
        BOOST_PP_CAT(BOOST_PP_TUPLE_ELEM(0, tuple), _info), GTREPO_as_string(BOOST_PP_TUPLE_ELEM(1, tuple)))
#define GTREPO_make_ctor_init_member_storage_infos(r, data, n, tuple) \
    BOOST_PP_COMMA_IF(n)                                              \
    BOOST_PP_CAT(GTREPO_fieldtype_name(tuple), _info)(BOOST_PP_CAT(GTREPO_fieldtype_name(tuple), _info))

// helper for ctor with dims
#define GTREPO_make_dim_args_helper(r, data, n, dim_value) BOOST_PP_COMMA_IF(n) BOOST_PP_CAT(dim, dim_value)
#define GTREPO_make_dim_args(tuple) \
    (BOOST_PP_LIST_FOR_EACH_I(GTREPO_make_dim_args_helper, ~, BOOST_PP_TUPLE_TO_LIST(tuple)))
#define GTREPO_make_ctor_with_dims_storage_infos(r, data, n, tuple) \
    BOOST_PP_COMMA_IF(n)                                            \
    BOOST_PP_CAT(GTREPO_fieldtype_name(tuple), _info)               \
    GTREPO_make_dim_args(GTREPO_fieldtype_dims_tuple(tuple))

#define GTREPO_init_map(r, data, tuple) \
    data_store_map_.emplace(            \
        GTREPO_as_string(BOOST_PP_TUPLE_ELEM(1, tuple)), GTREPO_name(BOOST_PP_TUPLE_ELEM(1, tuple)));

#define GTREPO_make_getters(r, data, tuple)                                               \
    BOOST_PP_TUPLE_ELEM(0, tuple) & BOOST_PP_CAT(get_, BOOST_PP_TUPLE_ELEM(1, tuple))() { \
        return GTREPO_name(BOOST_PP_TUPLE_ELEM(1, tuple));                                \
    }

#define GTREPO_is_data_store(r, data, fieldtype_tuple)                                        \
    GRIDTOOLS_STATIC_ASSERT((is_data_store< GTREPO_fieldtype_name(fieldtype_tuple) >::value), \
        "At least one of the arguments passed to the repository in GT_REPOSITORY_FIELDTYPES is not a data_store");

#define GTREPO_enum_of_fieldtypes_helper(r, data, n, tuple) BOOST_PP_COMMA_IF(n) GTREPO_fieldtype_name(tuple)
#define GTREPO_enum_of_fieldtypes(GTREPO_FIELDTYPES_SEQ) \
    BOOST_PP_SEQ_FOR_EACH_I(GTREPO_enum_of_fieldtypes_helper, ~, GTREPO_FIELDTYPES_SEQ)

#define GTREPO_max_fold_op(d, state, x) BOOST_PP_MAX_D(d, state, x)
#define GTREPO_max_in_tuple_fold(list) BOOST_PP_LIST_FOLD_LEFT(GTREPO_max_fold_op, 0, list)
#define GTREPO_max_in_tuple(tuple) GTREPO_max_in_tuple_fold(BOOST_PP_TUPLE_TO_LIST(tuple))

#define GTREPO_max_dim_fold_op(d, state, x) \
    BOOST_PP_MAX_D(d, state, GTREPO_max_in_tuple(GTREPO_fieldtype_dims_tuple(x)))
#define GTREPO_max_dim(GTREPO_FIELDTYPES_SEQ) BOOST_PP_SEQ_FOLD_LEFT(GTREPO_max_dim_fold_op, 0, GTREPO_FIELDTYPES_SEQ)

// non-zero if all fieldtypes have dimensions specified
#define GTREPO_has_dim_fold_op(d, state, x) BOOST_PP_MIN_D(d, state, BOOST_PP_DEC(BOOST_PP_TUPLE_SIZE(x)))
#define GTREPO_has_dim(GTREPO_FIELDTYPES_SEQ) \
    BOOST_PP_BOOL(BOOST_PP_SEQ_FOLD_LEFT(GTREPO_has_dim_fold_op, 1, GTREPO_FIELDTYPES_SEQ))

#define GTREPO_EMPTY(...)

#define GTREPO_make_dims_ctor(GTREPO_NAME, GTREPO_FIELDTYPES_SEQ, GTREPO_FIELDS_SEQ)                      \
    GTREPO_NAME(BOOST_PP_ENUM_PARAMS(BOOST_PP_ADD(GTREPO_max_dim(GTREPO_FIELDTYPES_SEQ), 1), uint_t dim)) \
        : BOOST_PP_SEQ_FOR_EACH_I(GTREPO_make_ctor_with_dims_storage_infos, ~, GTREPO_FIELDTYPES_SEQ),    \
          BOOST_PP_SEQ_FOR_EACH_I(GTREPO_make_ctor_init, ~, GTREPO_FIELDS_SEQ) {}

#define GTREPO_make_dims_ctor_if_has_dim(GTREPO_NAME, GTREPO_FIELDTYPES_SEQ, GTREPO_FIELDS_SEQ) \
    BOOST_PP_IF(GTREPO_has_dim(GTREPO_FIELDTYPES_SEQ),                                     \
        GTREPO_make_dims_ctor,GTREPO_EMPTY)(GTREPO_NAME, GTREPO_FIELDTYPES_SEQ, GTREPO_FIELDS_SEQ)

#define GTREPO_make_storage_info_ctor(GTREPO_NAME, GTREPO_FIELDTYPES_SEQ, GTREPO_FIELDS_SEQ)             \
    GTREPO_NAME(BOOST_PP_SEQ_FOR_EACH_I(GTREPO_make_ctor_args, ~, GTREPO_FIELDTYPES_SEQ))                \
        : BOOST_PP_SEQ_FOR_EACH_I(GTREPO_make_ctor_init_member_storage_infos, ~, GTREPO_FIELDTYPES_SEQ), \
          BOOST_PP_SEQ_FOR_EACH_I(GTREPO_make_ctor_init, ~, GTREPO_FIELDS_SEQ) {                         \
        BOOST_PP_SEQ_FOR_EACH(GTREPO_init_map, ~, GTREPO_FIELDS_SEQ)                                     \
    }

#define GTREPO_make_mpl_vector_of_fieldtypes(GTREPO_FIELDTYPES_SEQ) \
    using fieldtype_vector = boost::mpl::vector< GTREPO_enum_of_fieldtypes(GTREPO_FIELDTYPES_SEQ) >
// TODO add protection that all FieldTypes are present in GTREPO_FIELDTYPES_SEQ which are passed in GTREPOS_FIELDS_SEQ

#define GTREPO_is_valid_fieldtype(r, data, tuple)                                                              \
    GRIDTOOLS_STATIC_ASSERT((boost::mpl::contains< fieldtype_vector, GTREPO_field_type(tuple) >::type::value), \
        "At least one of the types in the fields sequence is not defined in the fieldtypes sequence.");

#define GT_MAKE_REPOSITORY_HELPER(GTREPO_NAME, GTREPO_FIELDTYPES_SEQ, GTREPO_FIELDS_SEQ)                      \
    class GTREPO_NAME {                                                                                       \
      private:                                                                                                \
        BOOST_PP_SEQ_FOR_EACH(GTREPO_make_storage_info_members, ~, GTREPO_FIELDTYPES_SEQ)                     \
        BOOST_PP_SEQ_FOR_EACH(GTREPO_make_members, ~, GTREPO_FIELDS_SEQ)                                      \
        std::unordered_map< std::string, boost::variant< GTREPO_enum_of_fieldtypes(GTREPO_FIELDTYPES_SEQ) > > \
            data_store_map_;                                                                                  \
        BOOST_PP_SEQ_FOR_EACH(GTREPO_is_data_store, ~, GTREPO_FIELDTYPES_SEQ)                                 \
        GTREPO_make_mpl_vector_of_fieldtypes(GTREPO_FIELDTYPES_SEQ);                                          \
        BOOST_PP_SEQ_FOR_EACH(GTREPO_is_valid_fieldtype, ~, GTREPO_FIELDS_SEQ)                                \
      public:                                                                                                 \
        GTREPO_make_storage_info_ctor(GTREPO_NAME, GTREPO_FIELDTYPES_SEQ, GTREPO_FIELDS_SEQ)                  \
            GTREPO_make_dims_ctor_if_has_dim(GTREPO_NAME, GTREPO_FIELDTYPES_SEQ, GTREPO_FIELDS_SEQ)           \
                GTREPO_NAME(const GTREPO_NAME &) = delete; /*non-copyable/moveable*/                          \
        GTREPO_NAME(GTREPO_NAME &&) = delete;                                                                 \
        BOOST_PP_SEQ_FOR_EACH(GTREPO_make_getters, ~, GTREPO_FIELDS_SEQ)                                      \
        auto data_stores() -> decltype(data_store_map_) & { return data_store_map_; }                         \
    };

#define GT_MAKE_REPOSITORY(GTREPO_NAME, GTREPO_FIELDTYPES_SEQ, GTREPO_FIELDS_SEQ) \
    GT_MAKE_REPOSITORY_HELPER(                                                    \
        GTREPO_NAME, PP_SEQ_DOUBLE_PARENS(GTREPO_FIELDTYPES_SEQ), PP_SEQ_DOUBLE_PARENS(GTREPO_FIELDS_SEQ))
