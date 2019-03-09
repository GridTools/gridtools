/*
 * GridTools
 *
 * Copyright (c) 2014-2019, ETH Zurich
 * All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */
#pragma once

#include <boost/fusion/include/vector.hpp>

#include "../common/defs.hpp"
#include "../common/host_device.hpp"
#include "../meta/always.hpp"
#include "../meta/list.hpp"
#include "./extent.hpp"
#include "./is_accessor.hpp"
#include "./is_global_accessor.hpp"

namespace gridtools {

    /** @brief internal struct to simplify the API when we pass arguments to the global_accessor ```operator()```

        \tparam GlobalAccessor the associated global_accessor
        \tparam Args the type of the arguments passed to the ```operator()``` of the global_accessor

        The purpose of this struct is to add a "state" to the global accessor, storing the arguments
        passed to it inside a tuple. The global_accessor_with_arguments is not explicitly instantiated by the user, it
       gets generated
        when calling the ```operator()``` on a global_accessor. Afterwards it is treated as an expression by
        the iterate_domain which contains an overload of ```operator()``` specialised for
       global_accessor_with_arguments.
     */
    template <typename GlobalAccessor, typename... Args>
    struct global_accessor_with_arguments {
      private:
        boost::fusion::vector<Args...> m_arguments;

      public:
        typedef typename GlobalAccessor::index_t index_t;
        static const constexpr intent intent_v = intent::in;

        GT_FUNCTION
        global_accessor_with_arguments(Args &&... args_) : m_arguments(std::forward<Args>(args_)...) {}
        GT_FUNCTION
        boost::fusion::vector<Args...> const &get_arguments() const { return m_arguments; };

        friend global_accessor_with_arguments tuple_getter(global_accessor_with_arguments const &) { return {}; }
        friend meta::list<> tuple_to_types(global_accessor_with_arguments const &) { return {}; }
        friend meta::always<global_accessor_with_arguments> tuple_from_types(global_accessor_with_arguments const &) {
            return {};
        }
    };

    template <typename Global, typename... Args>
    struct is_global_accessor<global_accessor_with_arguments<Global, Args...>> : std::true_type {};

    /**
       @brief Object to be accessed regardless of the current iteration point. A global_accessor is always read-only.

       \tparam I unique accessor identifier

       This accessor allows the user to call a user function contained in a user-defined object.
       Calling the parenthesis operator on the global_accessor generates an instance of
       ```global_accessor_with_arguments```.
     */
    template <uint_t I>
    struct global_accessor {

        static const constexpr intent intent_v = intent::in;

        typedef global_accessor<I> type;

        typedef static_uint<I> index_t;

        typedef extent<> extent_t;

        GT_FUNCTION constexpr global_accessor() {}

        /** @brief generates a global_accessor_with_arguments and returns it by value */
        template <typename... Args>
        GT_FUNCTION global_accessor_with_arguments<global_accessor, Args...> operator()(Args &&... args_) {
            return global_accessor_with_arguments<global_accessor, Args...>(std::forward<Args>(args_)...);
        }

        friend global_accessor tuple_getter(global_accessor const &) { return {}; }
        friend meta::list<> tuple_to_types(global_accessor const &) { return {}; }
        friend meta::always<global_accessor> tuple_from_types(global_accessor const &) { return {}; }
    };

    template <uint_t I>
    struct is_accessor<global_accessor<I>> : std::true_type {};

    template <uint_t I>
    struct is_global_accessor<global_accessor<I>> : std::true_type {};
} // namespace gridtools
