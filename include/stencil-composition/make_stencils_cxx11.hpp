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
    private:
        bool m_value;
    public:

        constexpr condition(){};

        template <uint_t ID>
        constexpr condition(conditional<ID> const& cond) : m_value(cond.value()) {
            GRIDTOOLS_STATIC_ASSERT((Tag::index_t::value==ID), "misformed input");
        }

        constexpr bool value() const {return m_value;}

        typedef Mss1 first;
        typedef Mss2 second;
        typedef Tag index_t;
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
    if_(Condition const& cond, Mss1&&, Mss2&&){
        GRIDTOOLS_STATIC_ASSERT(is_conditional<Condition>::value,
                                "you have to pass to gridtools::if_ an instance of type \"conditional\" as first argument.");
        return condition<Mss1, Mss2, Condition>(cond);
    }

    template <typename ... EsfDescr >
    independent_esf< boost::mpl::vector<EsfDescr ...> >
    make_independent(EsfDescr&& ... ) {
        return independent_esf<boost::mpl::vector<EsfDescr... > >();
    }

} // namespace gridtools
