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

#include <unordered_map>
#include "boost/variant.hpp"
#include "boost/preprocessor/seq.hpp"

#define PP_DETAIL_SEQ_DOUBLE_PARENS_0(...) ((__VA_ARGS__)) PP_DETAIL_SEQ_DOUBLE_PARENS_1
#define PP_DETAIL_SEQ_DOUBLE_PARENS_1(...) ((__VA_ARGS__)) PP_DETAIL_SEQ_DOUBLE_PARENS_0
#define PP_DETAIL_SEQ_DOUBLE_PARENS_0_END
#define PP_DETAIL_SEQ_DOUBLE_PARENS_1_END
#define PP_SEQ_DOUBLE_PARENS(seq) BOOST_PP_CAT(PP_DETAIL_SEQ_DOUBLE_PARENS_0 seq, _END)

#define GTREPO_name(name) BOOST_PP_CAT(name, _)
#define GTREPO_make_members(r, data, tuple) BOOST_PP_TUPLE_ELEM(0, tuple) GTREPO_name(BOOST_PP_TUPLE_ELEM(1, tuple));
#define GTREPO_as_string_helper(name) #name
#define GTREPO_as_string(name) GTREPO_as_string_helper(name)

#define GTREPO_make_ctor_args(r, data, n, data_store) \
    BOOST_PP_COMMA_IF(n) typename data_store::storage_info_t BOOST_PP_CAT(data_store, _info)
#define GTREPO_make_ctor_init(r, data, n, tuple) \
    BOOST_PP_COMMA_IF(n)                         \
    GTREPO_name(BOOST_PP_TUPLE_ELEM(1, tuple))(  \
        BOOST_PP_CAT(BOOST_PP_TUPLE_ELEM(0, tuple), _info), GTREPO_as_string(BOOST_PP_TUPLE_ELEM(1, tuple)))

#define GTREPO_init_map(r, data, tuple) \
    data_store_map_.emplace(            \
        GTREPO_as_string(BOOST_PP_TUPLE_ELEM(1, tuple)), GTREPO_name(BOOST_PP_TUPLE_ELEM(1, tuple)));

#define GTREPO_make_getters(r, data, tuple)                                               \
    BOOST_PP_TUPLE_ELEM(0, tuple) & BOOST_PP_CAT(get_, BOOST_PP_TUPLE_ELEM(1, tuple))() { \
        return GTREPO_name(BOOST_PP_TUPLE_ELEM(1, tuple));                                \
    }

#define GTREPO_is_data_store(r, data, data_store)                 \
    GRIDTOOLS_STATIC_ASSERT((is_data_store< data_store >::value), \
        "At least one of the arguments passed to the repository in GT_REPOSITORY_FIELDTYPES is not a data_store");

#define GT_MAKE_REPOSITORY_HELPER(GTREPO_NAME, GTREPO_FIELDTYPES_SEQ, GTREPO_FIELDS_SEQ)                               \
    class GTREPO_NAME {                                                                                                \
      private:                                                                                                         \
        BOOST_PP_SEQ_FOR_EACH(GTREPO_make_members, ~, GTREPO_FIELDS_SEQ)                                               \
        std::unordered_map< std::string, boost::variant< BOOST_PP_SEQ_ENUM(GTREPO_FIELDTYPES_SEQ) > > data_store_map_; \
        BOOST_PP_SEQ_FOR_EACH(GTREPO_is_data_store, ~, GTREPO_FIELDTYPES_SEQ)                                          \
      public:                                                                                                          \
        GTREPO_NAME(BOOST_PP_SEQ_FOR_EACH_I(GTREPO_make_ctor_args, ~, GTREPO_FIELDTYPES_SEQ))                          \
            : BOOST_PP_SEQ_FOR_EACH_I(GTREPO_make_ctor_init, ~, GTREPO_FIELDS_SEQ) {                                   \
            BOOST_PP_SEQ_FOR_EACH(GTREPO_init_map, ~, GTREPO_FIELDS_SEQ)                                               \
        }                                                                                                              \
        /*non-copyable/moveable*/                                                                                      \
        GTREPO_NAME(const GTREPO_NAME &) = delete;                                                                     \
        GTREPO_NAME(GTREPO_NAME &&) = delete;                                                                          \
        BOOST_PP_SEQ_FOR_EACH(GTREPO_make_getters, ~, GTREPO_FIELDS_SEQ)                                               \
        auto data_stores() -> decltype(data_store_map_) & { return data_store_map_; }                                  \
    };

#define GT_MAKE_REPOSITORY(GTREPO_NAME, GTREPO_FIELDTYPES_SEQ, GTREPO_FIELDS_SEQ) \
    GT_MAKE_REPOSITORY_HELPER(GTREPO_NAME, GTREPO_FIELDTYPES_SEQ, PP_SEQ_DOUBLE_PARENS(GTREPO_FIELDS_SEQ))
