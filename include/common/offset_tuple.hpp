#pragma once
#include <boost/mpl/or.hpp>
#include <boost/mpl/fold.hpp>
#include <boost/mpl/find.hpp>
#include "common/defs.hpp"
#include "stencil-composition/dimension_defs.hpp"
#include "common/generic_metafunctions/logical_ops.hpp"
#include "common/generic_metafunctions/variadic_to_vector.hpp"
#include "../common/generic_metafunctions/accumulate.hpp"
#include "common/generic_metafunctions/is_variadic_pack_of.hpp"
#include "../common/array.hpp"

namespace gridtools {

    /**@brief method for initializing the offsets in the placeholder
       Version valid for one dimension
       \param x is an instance of the \ref gridtools::enumtype::dimension class, which contains the offset (x.value) and
       the dimension index (X::direction)
    */
    template < ushort_t N, typename X >
    GT_FUNCTION constexpr int_t initialize(X x) {
        return (X::direction == N ? x.value : 0);
    }

#ifdef CXX11_ENABLED
    /**@brief method for initializing the offsets in the placeholder
       Version valid for arbitrary dimension
       \param x is an instance of the \ref gridtools::dimension class, which contains the offset (x.value) and the
       dimension index (X::direction)
       \param rest are the remaining arguments, which get considered one at a time in by means of recursive calls
    */
    template < ushort_t N, typename X, typename... Rest >
    GT_FUNCTION constexpr int_t initialize(X x, Rest... rest) {
        return X::direction == N ? x.value : initialize< N >(rest...);
    }
#else

    /**@brief method for initializing the offsets in the placeholder
       Version valid for two dimension
       \param x is an instance of the \ref gridtools::dimension class, which contains the offset (x.value) and the
       dimension index (X::direction)
       \param y is an instance of the \ref gridtools::dimension class, which contains the offset (y.value) and the
       dimension index (Y::direction)
    */
    template < ushort_t N, typename X, typename Y >
    GT_FUNCTION constexpr int_t initialize(X x, Y y) {
        return X::direction == N ? x.value : Y::direction == N ? y.value : 0;
    }

    /**@brief method for initializing the offsets in the placeholder
       Version valid for three dimension
       \param x is an instance of the \ref gridtools::dimension class, which contains the offset (x.value) and the
       dimension index (X::direction)
       \param y is an instance of the \ref gridtools::dimension class, which contains the offset (y.value) and the
       dimension index (Y::direction)
       \param z is an instance of the \ref gridtools::dimension class, which contains the offset (z.value) and the
       dimension index (Z::direction)
    */
    template < ushort_t N, typename X, typename Y, typename Z >
    GT_FUNCTION constexpr int_t initialize(X x, Y y, Z z) {
        return X::direction == N ? x.value : Y::direction == N ? y.value : Z::direction == N ? z.value : 0;
    }

    /**@brief method for initializing the offsets in the placeholder
       Version valid for three dimension
       \param x is an instance of the \ref gridtools::dimension class, which contains the offset (x.value) and the
       dimension index (X::direction)
       \param y is an instance of the \ref gridtools::dimension class, which contains the offset (y.value) and the
       dimension index (Y::direction)
       \param z is an instance of the \ref gridtools::dimension class, which contains the offset (z.value) and the
       dimension index (Z::direction)
    */
    template < ushort_t N, typename X, typename Y, typename Z, typename T >
    GT_FUNCTION constexpr int_t initialize(X x, Y y, Z z, T t) {
        return X::direction == N ? x.value : Y::direction == N ? y.value : Z::direction == N
                                                                               ? z.value
                                                                               : T::direction == N ? t.value : 0;
    }
#endif

    namespace _impl {
#ifdef CXX11_ENABLED
        template < typename... GenericElements >
        struct contains_array {
            typedef typename boost::mpl::fold<
                typename variadic_to_vector< typename is_array< GenericElements >::type... >::type,
                boost::mpl::false_,
                boost::mpl::or_< boost::mpl::_1, boost::mpl::_2 > >::type type;
        };
#endif
    }

    //################################################################################
    //                              Multidimensional Fields
    //################################################################################

    /**@brief this is a decorator of the arg_type, which is matching the extra dimensions
       \param n_args is the current ID of the extra dimension
       \param index_type is the index of the storage type

       EXAMPLE:

       Possible interfaces to access one extra dimension T (say, temperature) at offset -1 of a 'velocity' field V are
       the following:
       - specify it with the first integer argument (the arguments after the first define the offsets of the 3D fields
       and can be in any of the form described in gridtools::arg_type)
       \verbatim
       V(-1, x(1), z(-3));
       \endverbatim
       - specify explicitly the dimension: in this case the order of the arguments is arbitrary:
       \verbatim
       typedef dimension<4> T;
       V(x(1), z(-3), T(-1))
       \endverbatim

       Note that if no value is specified for the extra dimension a zero offset is implicitly assumed.
    */
    template < int_t Index, int_t NDim >
    struct offset_tuple : public offset_tuple< Index - 1, NDim > {
        static const int_t n_dim = NDim;

        typedef offset_tuple< Index - 1, NDim > super;
        static const short_t n_args = super::n_args + 1;

        GT_FUNCTION CONSTEXPR offset_tuple(const uint_t pos, array< int_t, NDim > const &offsets)
            : super(pos + 1, offsets), m_offset(offsets[pos]) {
#ifndef NDEBUG
            GTASSERT(pos < NDim);
#endif
        }
#ifdef CXX11_ENABLED

        /**@brief constructor taking an integer as the first argument, and then other optional arguments.
           The integer gets assigned to the current extra dimension and the other arguments are passed to the base
           class
           (in order to get assigned to the other dimensions).
           When this constructor is used all the arguments have to be specified and passed to the function call in
           order. No check is done on the order*/
        template < typename... GenericElements,
            typename =
                typename boost::disable_if< typename _impl::contains_array< GenericElements... >::type, bool >::type >
        GT_FUNCTION constexpr offset_tuple(int const t, GenericElements const... x)
            : super(x...), m_offset(t) {}

        /**@brief constructor taking the dimension class as argument.
           This allows to specify the extra arguments out of order. Note that 'dimension' is a
           language keyword used at the interface level.
        */
        template < ushort_t Idx, typename... GenericElements >
        GT_FUNCTION constexpr offset_tuple(dimension< Idx > const t, GenericElements const... x)
            : super(t, x...), m_offset(initialize< super::n_dim - n_args + 1 >(t, x...)) {}
#else
        /**@brief constructor taking an integer as the first argument, and then other optional arguments.
           The integer gets assigned to the current extra dimension and the other arguments are passed to the base
           class
           (in order to get assigned to the other dimensions).
           When this constructor is used all the arguments have to be specified and passed to the function call in
           order. No check is done on the order*/
        GT_FUNCTION offset_tuple(int const i, int const j, int const k) : super(j, k), m_offset(i) {}
        GT_FUNCTION
        offset_tuple(int const i, int const j) : super(j), m_offset(i) {}
        GT_FUNCTION
        offset_tuple(int const i) : m_offset(i) {}

        /**@brief constructor taking the Dimension class as argument.
           This allows to specify the extra arguments out of order. Note that 'enumtype::dimension' is a
           language keyword used at the interface level.
        */
        template < ushort_t Idx1, ushort_t Idx2, ushort_t Idx3, ushort_t Idx4 >
        GT_FUNCTION offset_tuple(
            dimension< Idx1 > const t, dimension< Idx2 > const u, dimension< Idx3 > const v, dimension< Idx4 > const &h)
            : super(t, u, v, h), m_offset(initialize< super::n_dim - n_args + 1 >(t, u, v, h)) {}

        /**@brief constructor taking the Dimension class as argument.
           This allows to specify the extra arguments out of order. Note that 'enumtype::dimension' is a
           language keyword used at the interface level.
        */
        template < ushort_t Idx1, ushort_t Idx2, ushort_t Idx3 >
        GT_FUNCTION offset_tuple(dimension< Idx1 > const t, dimension< Idx2 > const u, dimension< Idx3 > const v)
            : super(t, u, v), m_offset(initialize< super::n_dim - n_args + 1 >(t, u, v)) {}

        /**@brief constructor taking the Dimension class as argument.
           This allows to specify the extra arguments out of order. Note that 'dimension' is a
           language keyword used at the interface level.
        */
        template < ushort_t Idx1, ushort_t Idx2 >
        GT_FUNCTION offset_tuple(dimension< Idx1 > const t, dimension< Idx2 > const u)
            : super(t, u), m_offset(initialize< super::n_dim - n_args + 1 >(t, u)) {}

        /**@brief constructor taking the Dimension class as argument.
           This allows to specify the extra arguments out of order. Note that 'dimension' is a
           language keyword used at the interface level.
        */
        template < ushort_t Idx >
        GT_FUNCTION offset_tuple(dimension< Idx > const t)
            : super(t), m_offset(initialize< super::n_dim - n_args + 1 >(t)) {}
#endif

        // initializes recursively all the offsets to 0
        GT_FUNCTION
        constexpr offset_tuple() : super(), m_offset(0) {}

        template < short_t Idx >
        GT_FUNCTION constexpr bool end() const {
            return Idx == n_args - 1 ? false : super::template end< Idx >();
        }

        /**@brief returns the offset at a specific index Idx*/
        template < short_t Idx >
        /**@brief returns the offset array*/
        GT_FUNCTION constexpr int_t get() const {
            // NOTE: this should be a constexpr whenever m_offset is a static const
            // this might not be compile-time efficient for large indexes,
            // because both taken and not taken branches are compiled. boost::mpl::eval_if would be better.
            return Idx == n_args - 1 ? m_offset : super::template get< Idx >();
        }

      protected:
        int_t m_offset;
    };

    // specialization
    template < int_t NDim >
    struct offset_tuple< 0, NDim > {
        static const int_t n_dim = NDim;

        GT_FUNCTION CONSTEXPR offset_tuple(const uint_t pos, array< int_t, NDim > const &offsets) {
#ifndef NDEBUG
            assert(pos == NDim);
#endif
        }

#ifdef CXX11_ENABLED
        template < typename... GenericElements,
            typename =
                typename boost::disable_if< typename _impl::contains_array< GenericElements... >::type, bool >::type >
        GT_FUNCTION constexpr offset_tuple(GenericElements... x) {
            GRIDTOOLS_STATIC_ASSERT(is_variadic_pack_of(is_dimension< GenericElements >::type::value...),
                "wrong type for the argument of an offset_tuple");
        }

        // copy ctor
        GT_FUNCTION
        constexpr offset_tuple(const offset_tuple &other) {}
#else
        template < typename X, typename Y, typename Z, typename T >
        GT_FUNCTION constexpr offset_tuple(X x, Y y, Z z, T t) {}

        template < typename X, typename Y, typename Z >
        GT_FUNCTION constexpr offset_tuple(X x, Y y, Z z) {}

        template < typename X >
        GT_FUNCTION constexpr offset_tuple(X x) {}

        template < typename X, typename Y >
        GT_FUNCTION constexpr offset_tuple(X x, Y y) {}
#endif

        GT_FUNCTION
        constexpr offset_tuple() {}
        static const short_t n_args = 0;

        template < short_t Idx >
        GT_FUNCTION constexpr int_t get() const {
            return 0;
        }
    };

    template < typename T >
    struct is_offset_tuple : boost::mpl::false_ {};

    template < int_t Index, int_t NDim >
    struct is_offset_tuple< offset_tuple< Index, NDim > > : boost::mpl::true_ {};

#if defined(CXX11_ENABLED) && !defined(CUDA_CXX11_BUG_1) && !defined(__INTEL_COMPILER)

    /**
     * This provides equivalent functionality than offset_tuple, but when some dimension offsets
     * are provided as compile time information
     * @tparam Coordinates vector of integers marking the dimensions provided at compile time
     * @tparam NDim number of dimensions of the offset_tuple
     * @tparam DimensionPairs variadic of pairs (dimension, value) provided at compile time
     */
    template < typename Coordinates, int_t NDim, typename... DimensionPairs >
    struct offset_tuple_mixed {
        static const int_t n_dim = NDim;

        typedef offset_tuple< NDim, NDim > offset_tuple_t;

      private:
        const offset_tuple_t m_dynamic_offset_tuple;
        // compile time offset tuple
        static constexpr offset_tuple_t s_static_offset_tuple{
            dimension< DimensionPairs::first >{DimensionPairs::second}...};

      public:
        template < typename... ArgsRuntime >
        GT_FUNCTION constexpr offset_tuple_mixed(const ArgsRuntime ... args)
            : m_dynamic_offset_tuple(args...) {}

        template < short_t Idx >
        GT_FUNCTION constexpr int_t get() const {
            // we dispatch the getter to the static or dynamic depending on wether the dimension has been captured
            // compile time
            // (in the list of coordinates)
            return boost::is_same< typename boost::mpl::find< Coordinates, static_int< Idx > >::type,
                       typename boost::mpl::end< Coordinates >::type >::type::value
                       ? m_dynamic_offset_tuple.template get< Idx >()
                       : s_static_offset_tuple.template get< Idx >();
        }
    };

    template < typename Coordinates, int_t NDim, typename... Pair >
    constexpr const offset_tuple< NDim, NDim > offset_tuple_mixed< Coordinates, NDim, Pair... >::s_static_offset_tuple;

    template < typename Coordinates, int_t NDim, typename... DimensionPairs >
    struct is_offset_tuple< offset_tuple_mixed< Coordinates, NDim, DimensionPairs... > > : boost::mpl::true_ {};
#endif
} // namespace gridtools
