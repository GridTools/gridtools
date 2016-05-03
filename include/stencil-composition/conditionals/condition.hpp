#pragma once
#include "conditional.hpp"
/**@file*/
namespace gridtools {

    /**@brief structure containing a conditional and the two branches

       This structure is the record associated to a conditional, it contains two multi-stage stencils,
       possibly containing other conditionals themselves. One branch or the other will be eventually
       executed, depending on the content of the m_value member variable.
     */
    template < typename Mss1, typename Mss2, typename Tag >
    struct condition {

        GRIDTOOLS_STATIC_ASSERT(!(is_reduction_descriptor<Mss1>::value
                                  ||
                                  is_reduction_descriptor<Mss2>::value),
            "reduction mss must be outside conditional branches");
        // TODO add a way to check Mss1 and Mss2
        GRIDTOOLS_STATIC_ASSERT(is_conditional< Tag >::value, "internal error");
        typedef Mss1 first_t;
        typedef Mss2 second_t;
        typedef Tag index_t;

      private:
        index_t m_value;
        first_t m_first;
        second_t m_second;

      public:
        constexpr condition(){};

        constexpr condition(index_t cond, first_t const &first_, second_t const &second_)
            : m_value(cond), m_first(first_), m_second(second_) {}

        constexpr index_t value() const { return m_value; }
        constexpr second_t const &second() const { return m_second; }
        constexpr first_t const &first() const { return m_first; }
    };

    template < typename T >
    struct is_condition : boost::mpl::false_ {};

    template < typename Mss1, typename Mss2, typename Tag >
    struct is_condition< condition< Mss1, Mss2, Tag > > : boost::mpl::true_ {};
}
