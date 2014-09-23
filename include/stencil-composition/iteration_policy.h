#pragma once

namespace gridtools{
    namespace _impl{

/**\brief policy defining the behaviour on the vertical direction*/
        template<typename From, typename To, enumtype::execution ExecutionType>
        struct iteration_policy{};

/**\brief specialization for the forward iteration loop over k*/
        template<typename From, typename To>
        struct iteration_policy<From, To, enumtype::forward >
        {
            typedef From from;
            typedef To to;
            static inline int increment(int& k){return ++k;}
            static inline bool condition(int const& a, int const& b){return a<b;}
        };

/**\brief specialization for the backward iteration loop over k*/
        template<typename From,typename To>
        struct iteration_policy<From, To, enumtype::backward >
        {
            typedef  To from;
            typedef From to;
            static inline int increment(int& k){return --k;}
            static inline bool condition(int const& a, int const& b){return a>b;}
        };
    } // namespace _impl
} // namespace gridtools
