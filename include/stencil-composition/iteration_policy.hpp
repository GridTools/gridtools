#pragma once
#include "common/defs.hpp"

namespace gridtools{
    namespace _impl{

/**\brief policy defining the behaviour on the vertical direction*/
        template<typename From, typename To, enumtype::execution ExecutionType>
        struct iteration_policy{};

/**\brief specialization for the forward iteration loop over k*/
        template<typename From, typename To>
        struct iteration_policy<From, To, enumtype::forward >
        {
            static const enumtype::execution value=enumtype::forward;

            typedef From from;
            typedef To to;

            GT_FUNCTION
            static uint_t increment(uint_t& k){return ++k;}

            template<typename Domain>
            GT_FUNCTION
            static void increment(Domain& dom){dom.template increment<2, static_int<1> >();}

            GT_FUNCTION
            static bool condition(uint_t const& a, uint_t const& b){return a<=b;}//because the k dimension excludes the extremes, so we want to loop on the internal levels (otherwise we should have allocated more memory)
        };

/**\brief specialization for the backward iteration loop over k*/
        template<typename From,typename To>
        struct iteration_policy<From, To, enumtype::backward >
        {
            static const enumtype::execution value=enumtype::backward;
            typedef  To from;
            typedef From to;

            GT_FUNCTION
            static uint_t increment(uint_t& k){return --k;}

            template <typename Domain>
            GT_FUNCTION
            static void increment(Domain& dom){dom.template increment<2, static_int<-1> >();}

            GT_FUNCTION
            static bool condition(uint_t const& a, uint_t const& b){return a>=b;}//because the k dimension excludes the extremes, so we want to loop on the internal levels (otherwise we should have allocated more memory)
        };
    } // namespace _impl
} // namespace gridtools
