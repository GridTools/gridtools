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

#include "../../common/boost_pp_generic_macros.hpp"
#include "boost/preprocessor/seq.hpp"
#include <boost/preprocessor/tuple.hpp>
#include <boost/preprocessor/list.hpp>
#include <boost/preprocessor/selection/max.hpp>
#include <boost/preprocessor/selection/min.hpp>

/*
 * @brief data_store_types_tuple is a tuple of the form (DataStoreType, DimTuple). The following macros
 * provide named getters to the tuple elements.
 */
#define GTREPO_data_store_types_get_typename(data_store_types_tuple) BOOST_PP_TUPLE_ELEM(0, data_store_types_tuple)
#define GTREPO_data_store_types_get_dim_tuple(data_store_types_tuple) BOOST_PP_TUPLE_ELEM(1, data_store_types_tuple)
/*
 * @brief data_stores_tuple is a tuple of the form (DataStoreType, MemberName). The following macros
 * provide named getters to the tuple elements.
 */
#define GTREPO_data_stores_get_typename(data_stores_tuple) BOOST_PP_TUPLE_ELEM(0, data_stores_tuple)
#define GTREPO_data_stores_get_member_name(data_stores_tuple) BOOST_PP_TUPLE_ELEM(1, data_stores_tuple)
#define GTREPO_data_stores_get_member_name_underscore(data_stores_tuple) \
    BOOST_PP_CAT(BOOST_PP_TUPLE_ELEM(1, data_stores_tuple), _)

/*
 * GTREPO_max_in_tuple returns the maximum value in a BOOST_PP tuple
 */
#define GTREPO_max_fold_op(d, state, x) BOOST_PP_MAX_D(d, state, x)
#define GTREPO_max_in_tuple_fold(list) BOOST_PP_LIST_FOLD_LEFT(GTREPO_max_fold_op, 0, list)
#define GTREPO_max_in_tuple(tuple) GTREPO_max_in_tuple_fold(BOOST_PP_TUPLE_TO_LIST(tuple))

/*
 * GTREPO_max_dim returns the maximum dim in the DimensionTuple in the whole data_store_types sequence
 */
#define GTREPO_max_dim_fold_op(d, state, x) \
    BOOST_PP_MAX_D(d, state, GTREPO_max_in_tuple(GTREPO_data_store_types_get_dim_tuple(x)))
#define GTREPO_max_dim(data_store_types_seq) BOOST_PP_SEQ_FOLD_LEFT(GTREPO_max_dim_fold_op, 0, data_store_types_seq)

/*
 * @brief GTREPO_has_dim returns 0 if no dimensions are provided in at least one (DataStoreType, DimTuple) tuple
 */
#define GTREPO_has_dim_fold_op(d, state, x) BOOST_PP_MIN_D(d, state, BOOST_PP_DEC(BOOST_PP_TUPLE_SIZE(x)))
#define GTREPO_has_dim(data_store_types_seq) \
    BOOST_PP_BOOL(BOOST_PP_SEQ_FOLD_LEFT(GTREPO_has_dim_fold_op, 1, data_store_types_seq))
