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

            GT_FUNCTION
            static inline int increment(int& k){return ++k;}

            template<typename Domain>
            GT_FUNCTION
            static inline void increment(Domain& dom){dom.increment();}

            GT_FUNCTION
            static inline bool condition(int const& a, int const& b){return a+1<=b-1;}//because the k dimension excludes the extremes, so we want to loop on the internal levels (otherwise we should have allocated more memory)
        };

/**\brief specialization for the backward iteration loop over k*/
        template<typename From,typename To>
        struct iteration_policy<From, To, enumtype::backward >
        {
            typedef  To from;
            typedef From to;

            GT_FUNCTION
            static inline int increment(int& k){return --k;}

            template <typename Domain>
            GT_FUNCTION
            static inline void increment(Domain& dom){dom.decrement();}

            GT_FUNCTION
            static inline bool condition(int const& a, int const& b){return a-1>=b+1;}//because the k dimension excludes the extremes, so we want to loop on the internal levels (otherwise we should have allocated more memory)
        };
    } // namespace _impl
} // namespace gridtools
