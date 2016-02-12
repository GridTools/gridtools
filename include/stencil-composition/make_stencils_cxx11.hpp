#pragma once

#include "common/generic_metafunctions/variadic_to_vector.hpp"
#include "mss_metafunctions.hpp"
#include "mss.hpp"

namespace gridtools {

    template <typename T, typename Mss>
    struct case_type {
    private:
        T m_value;
        Mss m_mss;
    public:
        case_type(T val_, Mss mss_):
            m_value(val_),
            m_mss(mss_)
        {}

        Mss mss() const {return m_mss;}
        T value() const {return m_value;}
    };

    template <typename Mss>
    struct default_type{
    private:
        Mss m_mss;

    public:
        typedef Mss mss_t;

        default_type(Mss mss_):
            m_mss(mss_)
        {}

        Mss mss() const {return m_mss;}
    };

    template <typename T>
    struct is_case_type : boost::mpl::false_{};

    template <typename T, typename Mss>
    struct is_case_type<case_type<T, Mss> > : boost::mpl::true_{};

    template <typename T>
    struct is_default_type : boost::mpl::false_{};

    template <typename Mss>
    struct is_default_type<default_type<Mss> > : boost::mpl::true_{};

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

        //weak pointer
        int* m_value;

    public:
        typedef static_uint<Tag> index_t;
        static const uint_t index_value = index_t::value;

        constexpr conditional () //try to avoid this?
            : m_value()
        {}

        constexpr conditional (int*& c)
            : m_value(c)
        {}

        constexpr conditional (int& c)
            : m_value(&c)
        {}

        constexpr conditional (conditional const& other)
            : m_value(other.m_value)//shallow copy the pointer
        {}

        constexpr bool value() const {return *m_value;}
    };

    template <typename Mss1, typename Mss2, typename Tag>
    struct condition{

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


    template <uint_t Tag, typename T>
    class switch_variable{

        T m_value;
        uint_t m_num_cases;

    public:
        typedef static_uint<Tag> index_t;
        static const uint_t index_value = index_t::value;

        std::vector<int> m_conditions;

        constexpr switch_variable () //try to avoid this?
            : m_value()
            , m_conditions(0)
        {}

        constexpr switch_variable (T const& c)
            : m_value(c)
            , m_conditions(0)
        {}

        void push_back_condition( int c){m_conditions.push_back(c);}
        std::vector<int>& conditions( ){return m_conditions;}
        uint_t num_conditions( ){return m_conditions.size();}

        constexpr T value() const {return m_value;}
    };


    template <typename T>
    struct is_switch_variable : boost::mpl::false_ {};

    template <uint_t Tag, typename T>
    struct is_switch_variable<switch_variable<Tag, T> >:boost::mpl::true_ {};

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

    template<typename Condition, typename First, typename ... Cases>
    auto
    switch_(Condition& cond_, First const& first_, Cases ... cases_) ->
        decltype(
            if_(conditional<(uint_t) - (sizeof...(Cases)+sizeof...(Cases)*Condition::index_value)>()
                ,first_.mss()
                ,switch_(cond_, cases_ ...)
                )
            )
    {
        GRIDTOOLS_STATIC_ASSERT((is_case_type<First>::value),
                                "the entries in a switch_ statement must be case_ statements");

        //save the boolean in a vector owned by the switch variable
        //allows us to modify the switch at a later stage
        cond_.push_back_condition(cond_.value()==first_.value());
        //choose an ID which should be unique: to pick a very large number we cast a negative number to an unsigned
        int* cond_ptr = &cond_.m_conditions.back();
        conditional<(uint_t) - (sizeof...(Cases)+sizeof...(Cases)*Condition::index_value)> c(cond_ptr);
        return if_(c,
                   first_.mss(),
                   switch_(cond_, cases_ ...)
            );
    }


    template<typename Condition, typename Default>
    // typename switch_type<Condition, Default>::type
    typename Default::mss_t
    switch_(Condition const& cond_, Default const& last_){
        GRIDTOOLS_STATIC_ASSERT((is_default_type<Default>::value),
                                "the last entry in a switch_ statement must be a default_ statement");
        return last_.mss(); //default_ value
    }

    template<typename T, typename Mss>
    case_type<T, Mss> case_(T val_, Mss mss_)
    {
        return case_type<T, Mss>(val_, mss_);
    }

    template<typename Mss>
    default_type<Mss> default_(Mss mss_)
    {
        return default_type<Mss>(mss_);
    }

    template <typename ... EsfDescr >
    independent_esf< boost::mpl::vector<EsfDescr ...> >
    make_independent(EsfDescr&& ... ) {
        return independent_esf<boost::mpl::vector<EsfDescr... > >();
    }

} // namespace gridtools
