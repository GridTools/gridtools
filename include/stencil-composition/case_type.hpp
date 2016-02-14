#pragma once

namespace gridtools{
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
} // namespace gridtools
