#pragma once

#include "common/generic_metafunctions/variadic_to_vector.hpp"
#include "mss_metafunctions.hpp"
#include "mss.hpp"

namespace gridtools {

    template <typename ExecutionEngine,
        typename ... MssParameters >
    mss_descriptor<
        ExecutionEngine,
        typename extract_mss_esfs<typename variadic_to_vector<MssParameters ... >::type >::type,
        typename extract_mss_caches<typename variadic_to_vector<MssParameters ...>::type >::type
    >
    make_mss(ExecutionEngine&& /**/, MssParameters ...  ) {

        GRIDTOOLS_STATIC_ASSERT((is_execution_engine<ExecutionEngine>::value),
                                "The first argument passed to make_mss must be the execution engine (e.g. execute<forward>(), execute<backward>(), execute<parallel>()");

        return mss_descriptor<
            ExecutionEngine,
            typename extract_mss_esfs<typename variadic_to_vector<MssParameters ... >::type >::type,
            typename extract_mss_caches<typename variadic_to_vector<MssParameters ... >::type >::type
        >();
    }

    template <uint_t Tag>
    class conditional{

        bool m_value;

    public:
        typedef static_uint<Tag> index_t;
        static const uint_t index_value = index_t::value;

        constexpr conditional () //try to avoid this?
            : m_value(false)
        {}

        constexpr conditional (bool const& c)
            : m_value(c)
        {}
        constexpr bool value() const {return m_value;}
    };

    template <typename Mss1, typename Mss2, typename Tag>
    struct condition{

        typedef Mss1 first_t;
        typedef Mss2 second_t;
        typedef Tag index_t;

    private:
        bool m_value;
        first_t m_first;
        second_t m_second;
    public:

        constexpr condition(){};

        template <uint_t ID>
        constexpr condition(conditional<ID> const& cond, first_t const& first_, second_t const& second_)
            : m_value(cond.value())
            ,m_first(first_)
            ,m_second(second_)
        {
            GRIDTOOLS_STATIC_ASSERT((Tag::index_t::value==ID), "misformed input");
        }

        constexpr bool value() const {return m_value;}
        constexpr second_t const& second() const {return m_second;}
        constexpr first_t const& first() const {return m_first;}
    };

    template<typename T>
    struct is_condition: boost::mpl::false_{};

    template <typename Mss1, typename Mss2, typename Tag>
    struct is_condition<condition<Mss1, Mss2, Tag> >:boost::mpl::true_ {};

    template <typename T>
    struct is_conditional : boost::mpl::false_ {};

    template <uint_t Tag>
    struct is_conditional<conditional<Tag> >:boost::mpl::true_ {};

    template <typename Mss1, typename Mss2, typename Condition>
    condition<Mss1, Mss2, Condition>
    if_(Condition const& cond, Mss1 const& mss1_, Mss2 const& mss2_){
        GRIDTOOLS_STATIC_ASSERT(is_conditional<Condition>::value,
                                "you have to pass to gridtools::if_ an instance of type \"conditional\" as first argument.");
        return condition<Mss1, Mss2, Condition>(cond, mss1_, mss2_);
    }

    // template < typename FirstCondition, typename ... Conditions, typename First, typename ... Cases>
    // condition<First, switch_<Conditions ..., Cases ... >, FirstCondition >
    // switch_(Condition const& cond, Case&& ...){
    //     GRIDTOOLS_STATIC_ASSERT(is_conditional<Condition>::value,
    //                             "you have to pass to gridtools::if_ an instance of type \"conditional\" as first argument.");
    //     return condition<Mss1, Mss2, Condition>(cond);
    // }


    // template < typename Case, typename ... Mss>
    // case<Mss1, Mss2, Case>
    // case_(Case const& cond, Mss&& ...){
    //     GRIDTOOLS_STATIC_ASSERT(is_conditional<Condition>::value,
    //                             "you have to pass to gridtools::if_ an instance of type \"conditional\" as first argument.");
    //     return condition<Mss1, Mss2, Condition>(cond);
    // }


    template <typename ... EsfDescr >
    independent_esf< boost::mpl::vector<EsfDescr ...> >
    make_independent(EsfDescr&& ... ) {
        return independent_esf<boost::mpl::vector<EsfDescr... > >();
    }

} // namespace gridtools
