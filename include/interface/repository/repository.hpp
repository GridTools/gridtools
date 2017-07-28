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
#include "storage/storage-facility.hpp"
#include "boost/variant.hpp"

namespace gridtools {
// namespace {
//    template < typename T >
//    struct type_to_id {
//        GRIDTOOLS_STATIC_ASSERT(sizeof(T) < 0, GT_INTERNAL_ERROR);
//    };
//    template < typename T >
//    struct id_to_type {
//        GRIDTOOLS_STATIC_ASSERT(sizeof(T) < 0, GT_INTERNAL_ERROR);
//    };
//#define GT_REGISTER_FIELDTYPE(TYPE)                      \
//    static const int uniqueid_##TYPE = __COUNTER__;      \
//    template <>                                          \
//    struct type_to_id< TYPE > {                          \
//        const static int value = uniqueid_##TYPE;        \
//    };                                                   \
//    template <>                                          \
//    struct id_to_type< static_int< uniqueid_##TYPE > > { \
//        using type = TYPE;                               \
//    };
//#define GT_REGISTER_FIELD(TYPE, NAME)
//#include "my_repository.inc"
//#undef GT_REGISTER_FIELD
//#undef GT_REGISTER_FIELDTYPE
//}

#define FEL1(Action, X) Action(X)
#define FEL2(Action, X, ...) Action(X), FEL1(Action, __VA_ARGS__)
#define FEL3(Action, X, ...) Action(X), FEL2(Action, __VA_ARGS__)
#define FEL4(Action, X, ...) Action(X), FEL3(Action, __VA_ARGS__)
#define FEL5(Action, X, ...) Action(X), FEL4(Action, __VA_ARGS__)
#define FEL6(Action, X, ...) Action(X), FEL5(Action, __VA_ARGS__)
#define FEL7(Action, X, ...) Action(X), FEL6(Action, __VA_ARGS__)
#define FEL8(Action, X, ...) Action(X), FEL7(Action, __VA_ARGS__)
#define FEL9(Action, X, ...) Action(X), FEL8(Action, __VA_ARGS__)
#define FEL10(Action, X, ...) Action(X), FEL9(Action, __VA_ARGS__)
#define FEL11(Action, X, ...) Action(X), FEL10(Action, __VA_ARGS__)
#define FEL12(Action, X, ...) Action(X), FEL11(Action, __VA_ARGS__)

#define GET_MACRO(_1, _2, _3, _4, _5, _6, _7, _8, _9, _10, _11, _12, Name, ...) Name
#define FOR_EACH_LIST(Action, ...) \
    GET_MACRO(__VA_ARGS__, FEL12, FEL11, FEL10, FEL9, FEL8, FEL7, FEL6, FEL5, FEL4, FEL3, FEL2, FEL1)(Action,__VA_ARGS__)

    class GT_REPOSITORY_NAME {
      private:
#define GT_REGISTER_FIELD(Type, Name) Type m_##Name;
#define GT_REGISTER_FIELDTYPES(...)
#include GT_REPOSITORY_INC
#undef GT_REGISTER_FIELD
#undef GT_REGISTER_FIELDTYPES
        int dummy__; // HACK fix the trailing comma, see below

      public:
        std::unordered_map< std::string,
            boost::variant<
#define JustForward(X) X
#define GT_REGISTER_FIELD(Type, Name)
#define GT_REGISTER_FIELDTYPES(...) FOR_EACH_LIST(JustForward, __VA_ARGS__)
#include GT_REPOSITORY_INC
#undef GT_REGISTER_FIELD
#undef GT_REGISTER_FIELDTYPES
                                > > data_store_map_; // TODO fix the comma

#define GET_STORAGE_INFO(X) typename X::storage_info_t info_##X
        GT_REPOSITORY_NAME(
#define MakeCtor(Type) typename Type::storage_info_t info_##Type
#define GT_REGISTER_FIELD(Type, Name)
#define GT_REGISTER_FIELDTYPES(...) FOR_EACH_LIST(MakeCtor, __VA_ARGS__)
#include GT_REPOSITORY_INC
#undef GT_REGISTER_FIELD
#undef GT_REGISTER_FIELDTYPES
            )
            :
#define GT_REGISTER_FIELD(Type, Name) m_##Name(info_##Type),
#define GT_REGISTER_FIELDTYPES(...)
#include GT_REPOSITORY_INC
#undef GT_REGISTER_FIELD
#undef GT_REGISTER_FIELDTYPES
              dummy__(0) { // HACK fix the trailing comma
#define GT_REGISTER_FIELD(Type, Name) data_store_map_.emplace(#Name, m_##Name);
#define GT_REGISTER_FIELDTYPES(...)
#include GT_REPOSITORY_INC
#undef GT_REGISTER_FIELD
#undef GT_REGISTER_FIELDTYPES
        }

        /// @brief Configuration is non-copyable/moveable
        GT_REPOSITORY_NAME(const GT_REPOSITORY_NAME &) = delete;
        GT_REPOSITORY_NAME(GT_REPOSITORY_NAME &&) = delete;

// Getter & Setter
#define GT_REGISTER_FIELD(Type, Name)                                  \
    const Type &get_##Name() const noexcept { return this->m_##Name; } \
    void set_##Name(const Type &value) noexcept { this->m_##Name = value; }
#define GT_REGISTER_FIELDTYPES(...)
#include GT_REPOSITORY_INC
#undef GT_REGISTER_FIELD
#undef GT_REGISTER_FIELDTYPES

        auto data_stores() -> decltype(data_store_map_) & { return data_store_map_; }
    };
}

#undef GT_REPOSITORY_NAME
#undef GT_REPOSITORY_INC
