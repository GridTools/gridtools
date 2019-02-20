/*
 * GridTools
 *
 * Copyright (c) 2019, ETH Zurich
 * All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */

#if (!defined(BOOST_PP_VARIADICS) || (BOOST_PP_VARIADICS < 1))
// defining BOOST_PP_VARIADICS 1 here might be too late, therefore we leave it to the user
#error \
    "GRIDTOOLS ERROR=> For the repository you need to \"#define BOOST_PP_VARIADICS 1\" before the first include of any boost preprocessor file.")
#endif

#include "../../common/boost_pp_generic_macros.hpp"
#include "boost/preprocessor/seq.hpp"
#include <boost/preprocessor/list.hpp>
#include <boost/preprocessor/selection/max.hpp>
#include <boost/preprocessor/selection/min.hpp>
#include <boost/preprocessor/tuple.hpp>

/*
 * @brief data_store_types_tuple is a tuple of the form (DataStoreType, DimTuple). The following macros
 * provide named getters to the tuple elements.
 */
#define GT_REPO_data_store_types_get_typename(data_store_types_tuple) BOOST_PP_TUPLE_ELEM(0, data_store_types_tuple)
#define GT_REPO_data_store_types_get_dim_tuple(data_store_types_tuple) BOOST_PP_TUPLE_ELEM(1, data_store_types_tuple)
/*
 * @brief data_stores_tuple is a tuple of the form (DataStoreType, MemberName). The following macros
 * provide named getters to the tuple elements.
 */
#define GT_REPO_data_stores_get_typename(data_stores_tuple) BOOST_PP_TUPLE_ELEM(0, data_stores_tuple)
#define GT_REPO_data_stores_get_member_name(data_stores_tuple) BOOST_PP_TUPLE_ELEM(1, data_stores_tuple)
#define GT_REPO_data_stores_get_member_name_underscore(data_stores_tuple) \
    BOOST_PP_CAT(BOOST_PP_TUPLE_ELEM(1, data_stores_tuple), _)

/*
 * GT_REPO_max_in_tuple returns the maximum value in a BOOST_PP tuple
 */
#define GT_REPO_max_fold_op(d, state, x) BOOST_PP_MAX_D(d, state, x)
#define GT_REPO_max_in_tuple_fold(list) BOOST_PP_LIST_FOLD_LEFT(GT_REPO_max_fold_op, 0, list)
#define GT_REPO_max_in_tuple(tuple) GT_REPO_max_in_tuple_fold(BOOST_PP_TUPLE_TO_LIST(tuple))

/*
 * GT_REPO_max_dim returns the maximum dim in the DimensionTuple in the whole data_store_types sequence
 */
#define GT_REPO_max_dim_fold_op(d, state, x) \
    BOOST_PP_MAX_D(d, state, GT_REPO_max_in_tuple(GT_REPO_data_store_types_get_dim_tuple(x)))
#define GT_REPO_max_dim(data_store_types_seq) BOOST_PP_SEQ_FOLD_LEFT(GT_REPO_max_dim_fold_op, 0, data_store_types_seq)

/*
 * @brief GT_REPO_has_dim returns 0 if no dimensions are provided in at least one (DataStoreType, DimTuple) tuple
 */
#define GT_REPO_has_dim_fold_op(d, state, x) BOOST_PP_MIN_D(d, state, BOOST_PP_DEC(BOOST_PP_TUPLE_SIZE(x)))
#define GT_REPO_has_dim(data_store_types_seq) \
    BOOST_PP_BOOL(BOOST_PP_SEQ_FOLD_LEFT(GT_REPO_has_dim_fold_op, 1, data_store_types_seq))
