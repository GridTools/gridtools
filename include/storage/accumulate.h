namespace gridtools{

template <typename T1, typename T2  >
    struct max
    {
	static const int_t value = (T1::value > T2::value)? T1::value : T2::value;
	typedef static_int<value> type;
    };

    template<typename Vector, uint_t ID>
    struct find_max
    {
	typedef typename max< typename boost::mpl::at_c< Vector, ID>::type , typename find_max<Vector, ID-1>::type >::type type;
    };

    template<typename Vector>
    struct find_max<Vector, 0>
    {
	typedef typename boost::mpl::at_c<Vector, 0>::type type;
    };

    template<typename Vector>
    struct vec_max
    {
	typedef typename find_max< Vector, boost::mpl::size<Vector>::type::value-1>::type type;
	static const int_t value=find_max< Vector, boost::mpl::size<Vector>::type::value-1>::type::value;
    };

    struct multiplies {
	GT_FUNCTION
	constexpr multiplies(){}
	template <typename  T>
	GT_FUNCTION
	constexpr T operator() (const T& x, const T& y) const {return x*y;}
    };

    struct add {
	GT_FUNCTION
	constexpr add(){}
	template <class T>
	GT_FUNCTION
	constexpr T operator() (const T& x, const T& y) const {return x+y;}
    };

    template<typename Operator, typename First, typename ... Args>
    GT_FUNCTION
    static constexpr First accumulate(Operator op, First first, Args ... args ) {
	return op(first,accumulate(op, args ...));
    }

    template<typename Operator, typename First>
    GT_FUNCTION
    static constexpr First accumulate(Operator op, First first){return first;}
}//namespace gridtools
