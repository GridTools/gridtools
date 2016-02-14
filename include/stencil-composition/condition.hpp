#pragma once
#include "conditional.hpp"

namespace gridtools{
    template <typename Mss1, typename Mss2, typename Tag>
    struct condition{

        GRIDTOOLS_STATIC_ASSERT(is_conditional<Tag>::value, "internal error");
        typedef Mss1 first_t;
        typedef Mss2 second_t;
        typedef Tag index_t;

    private:
        index_t m_value;
        first_t m_first;
        second_t m_second;
    public:

        constexpr condition(){};

        template <uint_t ID>
        constexpr condition(conditional<ID> const& cond, first_t const& first_, second_t const& second_)
            : m_value(cond)
            ,m_first(first_)
            ,m_second(second_)
        {
        }

        constexpr index_t value() const {return m_value;}
        constexpr second_t const& second() const {return m_second;}
        constexpr first_t const& first() const {return m_first;}
    };

    template<typename T>
    struct is_condition: boost::mpl::false_{};

    template <typename Mss1, typename Mss2, typename Tag>
    struct is_condition<condition<Mss1, Mss2, Tag> >:boost::mpl::true_ {};
}
