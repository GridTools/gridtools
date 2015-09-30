#pragma once

#include "../common/defs.hpp"

namespace gridtools {
    namespace _impl {
        /**
           Function to perform positional_iterate_domain reset of iteration indices
           to be overhead-free for non positional computations
         */
        template <ushort_t Index, typename IterateDomain, typename VT>
        typename boost::enable_if<typename is_positional_iterate_domain<IterateDomain>::type, void>::type
        GT_FUNCTION
        reset_index_if_positional(IterateDomain & itdom, VT value) {
            itdom.template reset_index<Index>(value);
        }

        /**
           Function to perform positional_iterate_domain reset of iteration indices
           to be overhead-free for non positional computations.

           Overload for non positional computations
         */
        template <ushort_t Index, typename IterateDomain, typename VT>
        typename boost::disable_if<typename is_positional_iterate_domain<IterateDomain>::type, void>::type
        GT_FUNCTION
        reset_index_if_positional(IterateDomain &, VT) { }
    } // namespace _impl
} // namespace gridtools

