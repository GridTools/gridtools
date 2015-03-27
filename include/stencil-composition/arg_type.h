#pragma once
#include "arg_type_impl.h"
/**
   @file
   @brief File containing the definition of the placeholders used to address the storage from whithin the functors.
   A placeholder is an implementation of the proxy design pattern for the storage class, i.e. it is a light object used in place of the storage when defining the high level computations,
   and it will be bound later on with a specific instantiation of a storage class.

   Two different types of placeholders are considered:
   - arg represents the storage in the body of the main function, and it gets lazily assigned to a real storage.
   - arg_type represents the storage inside the functor struct containing a Do method. It can be instantiated directly in the Do method, or it might be a constant expression instantiated outside the functor scope and with static duration.
*/

namespace gridtools {

    /**
       @brief the definition of arg_type visible to the user

       \tparam ID the integer unic ID of the field placeholder
       \tparam Range the range of i/j indices spanned by the placeholder, in the form of <i_minus, i_plus, j_minus, j_plus>.
               The values are relative to the current position. See e.g. horizontal_diffusion::out_function as a usage example.
       \tparam Number the number of dimensions accessed by the field. Notice that we don't distinguish at this level what we call
               "space dimensions" from the "field dimensions". Both of them are accessed using the same interface. whether they are
               field dimensions or space dimension will be decided at the moment of the storage instantiation (in the main function)
     */
#ifdef CXX11_ENABLED
    template < ushort_t ID, typename Range=range<0,0,0,0>, ushort_t Number=3>
    using arg_type = typename arg_extend< ID, Range, Number, Number>::type;
#else
    template < ushort_t ID, typename Range=range<0,0,0,0>, ushort_t Number=3>
    struct arg_type {
        typedef typename arg_extend<ID, Range, Number, Number>::type type;
    };
#endif
    /**
     * Type to create placeholders for data fields.
     *
     * There is a specialization for the case in which T is a temporary
     *
     * @tparam I Integer index (unique) of the data field to identify it
     * @tparam T The type of the storage used to store data
     */
    template <uint_t I, typename T>
    struct arg {
        typedef T storage_type;
        typedef typename T::iterator_type iterator_type;
        typedef typename T::value_type value_type;
        typedef static_uint<I> index_type;
        typedef static_uint<I> index;

        template<typename Storage>
        arg_storage_pair<arg<I,T>, Storage>
        operator=(Storage& ref) {
            BOOST_MPL_ASSERT( (boost::is_same<Storage, T>) );
            return arg_storage_pair<arg<I,T>, Storage>(&ref);
        }

        static void info() {
            std::cout << "Arg on real storage with index " << I;
        }
    };

    namespace enumtype
    {
        /**
           @brief The following struct defines one specific component of a field
           It contains a direction (compile time constant, specifying the ID of the component), and a value
           (runtime value, which is storing the offset in the given direction).
           As everything what is inside the enumtype namespace, the Dimension keyword is supposed to be used at the application interface level.
        */
        template <ushort_t Coordinate>
        struct Dimension{
	    template <typename IntType>
            GT_FUNCTION
            constexpr Dimension(IntType val) : value
#if( (!defined(CXX11_ENABLED)) )
                                             (val)
#else
                {val}
#endif
                {
                    GRIDTOOLS_STATIC_ASSERT(Coordinate!=0, "The coordinate values passed to the arg_type start from 1")
                    GRIDTOOLS_STATIC_ASSERT(Coordinate>0, "The coordinate values passed to the arg_type must be positive integerts")
                }

	    GT_FUNCTION
	    constexpr Dimension(Dimension const& other):value(other.value){}

            static const ushort_t direction=Coordinate;
            int_t value;
	    struct Index{
		GT_FUNCTION
		Index(){}
		GT_FUNCTION
		Index(Index const&){}
		typedef Dimension<Coordinate> super;
	    };

	private:
	    Dimension();
        };

        /**Aliases for the first three dimensions (x,y,z)*/
        typedef Dimension<1> x;
        typedef Dimension<2> y;
        typedef Dimension<3> z;

    }

#ifdef CXX11_ENABLED

    /**TODO Document me*/
    template <typename ArgType, typename ... Pair>
    struct arg_mixed{

        static const ushort_t n_args = ArgType::n_args;
        static const ushort_t n_dim = ArgType::n_dim;
        typedef typename ArgType::base_t base_t;
        typedef typename ArgType::index_type index_type;
    private:
        static constexpr typename arg_extend<ArgType::index_type::value, typename ArgType::range_type, ArgType::n_dim, ArgType::n_dim>::type s_args_constexpr{ enumtype::Dimension<Pair::first>{Pair::second} ... };

        typename arg_extend<ArgType::index_type::value, typename ArgType::range_type, ArgType::n_dim, ArgType::n_dim>::type m_args_runtime;
    public:

        template<typename ... ArgsRuntime>
        GT_FUNCTION
        constexpr
        arg_mixed( ArgsRuntime const& ... args ): m_args_runtime(args...) {
        }

        /**@brief returns the offset at a specific index Idx*/
        template<short_t Idx>
        GT_FUNCTION
        static constexpr
        uint_t const&
        get_constexpr(){
            GRIDTOOLS_STATIC_ASSERT(Idx<s_args_constexpr.n_dim, "the idx must be smaller than the arg dimension")
            GRIDTOOLS_STATIC_ASSERT(Idx>=0, "the idx must be larger than 0")
            GRIDTOOLS_STATIC_ASSERT(s_args_constexpr.template get<Idx>()>=0, "the result must be larger or equal than 0")
            //     static_int<s_args_constexpr.template get<Idx>()>::fuck();
            // static_int<Idx>::fuck();
            return s_args_constexpr.template get<Idx>();
        }

        /**@brief returns the offset at a specific index Idx*/
        template<short_t Idx>
        GT_FUNCTION
        constexpr
        const int_t & get() const {
            return s_args_constexpr.template end<Idx>() ? s_args_constexpr.get<Idx>() : m_args_runtime.get<Idx>();
        }
    };

    template <typename ArgType, typename ... Pair>
    constexpr typename arg_extend<ArgType::index_type::value, typename ArgType::range_type, ArgType::n_dim, ArgType::n_dim>::type arg_mixed<ArgType, Pair...>::s_args_constexpr;


/**this struct allows the specification of SOME of the arguments before instantiating the arg_decorator.
   It is a language keyword.
*/
    template <typename Callable, typename ... Known>
    struct alias{

        template <int Arg1, int Arg2> struct pair_
        {
            static const int first=Arg1;
            static const int second=Arg2;
        };

        //compile-time aliases, the offsets specified in this way are assured to be compile-time
        template<int ... Args>
        using set=arg_mixed< Callable, pair_<Known::direction,Args> ... >;

        /**@brief constructor
       \param args are the offsets which are already known*/
        template<typename ... Args>
        GT_FUNCTION
        constexpr alias( Args/*&&*/ ... args ): m_knowns{args ...} {
        }

        typedef boost::mpl::vector<Known...> dim_vector;

        /** @brief operator calls the constructor of the arg_decorator
            \param unknowns are the parameters which were not known beforehand. They might be instances of
            the enumtype::Dimension class. Together with the m_knowns offsets form the arguments to be
            passed to the Callable functor (which is normally an instance of arg_decorator)
        */
    template<typename ... Unknowns>
    GT_FUNCTION
    Callable/*&&*/ operator() ( Unknowns/*&&*/ ... unknowns  ) const
            {
                return Callable(enumtype::Dimension<Known::direction> (m_knowns[boost::mpl::find<dim_vector, Known>::type::pos::value]) ... , unknowns ...);}

    private:
        //store the list of offsets which are already known on an array
        int_t m_knowns [sizeof...(Known)];
    };
#endif

} // namespace gridtools
