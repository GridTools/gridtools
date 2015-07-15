#pragma once
#include "accessor_impl.hpp"
/**
   @file
   @brief File containing the definition of the placeholders used to address the storage from whithin the functors.
   A placeholder is an implementation of the proxy design pattern for the storage class, i.e. it is a light object used in place of the storage when defining the high level computations,
   and it will be bound later on with a specific instantiation of a storage class.

   Two different types of placeholders are considered:
   - arg represents the storage in the body of the main function, and it gets lazily assigned to a real storage.
   - accessor represents the storage inside the functor struct containing a Do method. It can be instantiated directly in the Do method, or it might be a constant expression instantiated outside the functor scope and with static duration.
*/

namespace gridtools {

    /**
       @brief the definition of accessor visible to the user

       \tparam ID the integer unic ID of the field placeholder
       \tparam Range the range of i/j indices spanned by the placeholder, in the form of <i_minus, i_plus, j_minus, j_plus>.
               The values are relative to the current position. See e.g. horizontal_diffusion::out_function as a usage example.
       \tparam Number the number of dimensions accessed by the field. Notice that we don't distinguish at this level what we call
       "space dimensions" from the "field dimensions". Both of them are accessed using the same interface. whether they are
       field dimensions or space dimension will be decided at the moment of the storage instantiation (in the main function)
     */
    template < uint_t ID, typename Range=range<0,0,0,0>, ushort_t Number=3>
    struct accessor : public accessor_base<ID, Range, Number> {
        typedef accessor_base<ID, Range, Number> super;
#ifdef CXX11_ENABLED

        GT_FUNCTION
        constexpr accessor(): super() {}

#ifndef __CUDACC__
        /**inheriting all constructors from offset_tuple*/
        using super::accessor_base;
#else
        /**@brief constructor forwarding all the arguments
        */
        template <typename... ForwardedArgs>
        GT_FUNCTION
        constexpr accessor ( ForwardedArgs... x): super (x...) {}

        //move ctor
        GT_FUNCTION
        constexpr explicit accessor(accessor<ID, Range, Number>&& other) : super(std::move(other)) {}

        //copy ctor
        GT_FUNCTION
        constexpr accessor(accessor<ID, Range, Number> const& other) : super(other) {
        }
#endif
#else

        //copy ctor
        GT_FUNCTION
        constexpr explicit accessor(accessor<ID, Range, Number> const& other) : super(other) {}

        //copy ctor from an accessor with different ID
        template<ushort_t OtherID>
        GT_FUNCTION
        constexpr explicit accessor(const accessor<OtherID, Range, Number>& other) :
            super(static_cast<accessor_base<OtherID, Range, Number> >(other)) {}

        GT_FUNCTION
        constexpr explicit accessor(): super() {}

        /** @brief constructor forwarding all the arguments*/
        template <typename X, typename Y, typename Z,  typename T>
        GT_FUNCTION
        constexpr accessor ( X x, Y y, Z z, T t ): super(x, y, z, t) {}

        /** @brief constructor forwarding all the arguments*/
        template <typename X, typename Y, typename Z>
        GT_FUNCTION
        constexpr accessor ( X x, Y y, Z z ): super(x, y, z) {}

        /** @brief constructor forwarding all the arguments*/
        template <typename X>
        GT_FUNCTION
        constexpr accessor ( X x ) : super(x) {}

        /** @brief constructor forwarding all the arguments*/
        template <typename X, typename Y>
        GT_FUNCTION
        constexpr accessor ( X x, Y y ): super(x, y) {}

#endif
    };

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
            GRIDTOOLS_STATIC_ASSERT( (boost::is_same<Storage, T>::value), "there is a mismatch between the storage types used by the arg placeholders and the storages really instantiated. Check that the placeholders you used when constructing the domain_type are in the correctly assigned and that their type match the instantiated storages ones" );


            return arg_storage_pair<arg<I,T>, Storage>(&ref);
        }

        static void info() {
            std::cout << "Arg on real storage with index " << I;
        }
    };

    template<typename T>
    struct is_arg : boost::mpl::false_{};

    template<uint_t I, typename Storage>
    struct is_arg<arg<I, Storage> > : boost::mpl::true_{};

    namespace enumtype
    {
        /**
           @section enumtype
           @{
           @brief The following struct defines one specific component of a field
           It contains a direction (compile time constant, specifying the ID of the component),
           and a value (runtime value, which is storing the offset in the given direction).
           As everything what is inside the enumtype namespace, the Dimension keyword is
           supposed to be used at the application interface level.
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
                GRIDTOOLS_STATIC_ASSERT(Coordinate!=0, "The coordinate values passed to the accessor start from 1");
                GRIDTOOLS_STATIC_ASSERT(Coordinate>0, "The coordinate values passed to the accessor must be positive integerts");
            }

            /**@brief Constructor*/
            GT_FUNCTION
            constexpr Dimension(Dimension const& other):value(other.value){}

            static const ushort_t direction=Coordinate;
            int_t value;

            /**@brief syntactic sugar for user interface

               overloaded operators return Index types which provide the proper Dimension object.
               Clarifying example:
               defining
               \code{.cpp}
               typedef Dimension<5>::Index t;
               \endcode
               we can use thefollowing alias
               \code{.cpp}
               t+2 <--> Dimension<5>(2)
               \endcode

             */
            struct Index{
               GT_FUNCTION
               constexpr Index(){}
                GT_FUNCTION
                constexpr Index(Index const&){}
                typedef Dimension<Coordinate> super;
            };

        private:
            Dimension();
        };

        /**Aliases for the first three dimensions (x,y,z)*/
        typedef Dimension<1> x;
        typedef Dimension<2> y;
        typedef Dimension<3> z;

        /**@}*/
    }

#if defined(CXX11_ENABLED) && !defined(__CUDACC__)

    /**@brief same as accessor but mixing run-time offsets with compile-time ones

       When we know beforehand that the dimension which we are querying is
       a compile-time one, we can use the static method get_constexpr() to get the offset.
       Otherwise the method get() checks before among the static dimensions, and if the
       queried dimension is not found it looks up in the dynamic dimensions. Note that this
       lookup is anyway done at compile time, i.e. the get() method returns in constant time.
     */
    template <typename ArgType, typename ... Pair>
    struct accessor_mixed{

        static const ushort_t n_dim = ArgType::n_dim;
        typedef typename ArgType::base_t base_t;
        typedef typename ArgType::index_type index_type;
    private:
        static constexpr accessor_base<ArgType::index_type::value
                                             , typename ArgType::range_type
                                             , ArgType::n_dim> s_args_constexpr{
            enumtype::Dimension<Pair::first>{Pair::second} ... };

        accessor_base<ArgType::index_type::value
                      , typename ArgType::range_type
                      , ArgType::n_dim> m_args_runtime;
        typedef boost::mpl::vector<static_int<n_dim-Pair::first> ... > coordinates;
    public:

        template<typename ... ArgsRuntime>
        GT_FUNCTION
        constexpr
        accessor_mixed( ArgsRuntime const& ... args ): m_args_runtime(args...) {
        }

        /**@brief returns the offset at a specific index Idx

           this is the constexpr version of the get() method (e.g. can be used as template parameter).
         */
        template<short_t Idx>
        GT_FUNCTION
        static constexpr
        uint_t const
        get_constexpr(){
            GRIDTOOLS_STATIC_ASSERT(Idx<s_args_constexpr.n_dim, "the idx must be smaller than the arg dimension");
            GRIDTOOLS_STATIC_ASSERT(Idx>=0, "the idx must be larger than 0");

            GRIDTOOLS_STATIC_ASSERT(s_args_constexpr.template get<Idx>()>=0, "there is a negative offset. If you did this on purpose recompile with the PEDANTIC_DISABLED flag on.");
            return s_args_constexpr.template get<Idx>();
        }

        /**@brief returns the offset at a specific index Idx

           the lookup for the index Idx is done at compile time, i.e. this method returns in constant time
         */
        template<short_t Idx>
        GT_FUNCTION
        constexpr
        const int_t get() const {
            return boost::is_same<typename boost::mpl::find<coordinates, static_int<Idx> >::type, typename boost::mpl::end<coordinates>::type >::type::value ? m_args_runtime.template get<Idx>() : s_args_constexpr.template get<Idx>() ;
        }
    };

    template <typename ArgType, typename ... Pair>
    constexpr accessor_base<ArgType::index_type::value
                                  , typename ArgType::range_type
                                  , ArgType::n_dim> accessor_mixed<ArgType, Pair...>::s_args_constexpr;


/**this struct allows the specification of SOME of the arguments before instantiating the offset_tuple.
   It is a language keyword.
*/
    template <typename Callable, typename ... Known>
    struct alias{

        template <int Arg1, int Arg2> struct pair_
        {
            static const int first=Arg1;
            static const int second=Arg2;
        };

        /**
           @brief compile-time aliases, the offsets specified in this way are assured to be compile-time

           This type alias allows to embed some of the offsets directly inside the type of the accessor placeholder.
           For a usage example check the exaples folder
        */
        template<int ... Args>
        using set=accessor_mixed< Callable, pair_<Known::direction,Args> ... >;

        /**@brief constructor
       \param args are the offsets which are already known*/
        template<typename ... Args>
        GT_FUNCTION
        constexpr alias( Args/*&&*/ ... args ): m_knowns{args ...} {
        }

        typedef boost::mpl::vector<Known...> dim_vector;

        /** @brief operator calls the constructor of the offset_tuple
            \param unknowns are the parameters which were not known beforehand. They might be instances of
            the enumtype::Dimension class. Together with the m_knowns offsets form the arguments to be
            passed to the Callable functor (which is normally an instance of offset_tuple)
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
