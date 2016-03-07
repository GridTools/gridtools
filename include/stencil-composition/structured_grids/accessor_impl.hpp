#pragma once

#include "storage/storage.hpp"
#include "common/layout_map.hpp"
#include <boost/type_traits/integral_constant.hpp>
#include <boost/mpl/assert.hpp>
#include <boost/mpl/for_each.hpp>
#include <boost/fusion/container/vector.hpp>
#include <boost/fusion/include/for_each.hpp>
#include <vector>
#include "common/is_temporary_storage.hpp"
#include "stencil-composition/offset_tuple.hpp"
#include "storage/storage_metafunctions.hpp"
#include "storage/storage.hpp"
#include "common/layout_map.hpp"
#include "stencil-composition/extent.hpp"

#ifdef CXX11_ENABLED
#include "../expressions/expressions.hpp"
#endif

namespace gridtools {

    //forward declaration
    template< int_t  Index, int_t NDim >
    struct offset_tuple;

    template <ushort_t>
    struct dimension;

    template <uint_t I, typename T, typename Cond>
    struct arg;

    /**
     * @brief Type to be used in elementary stencil functions to specify argument mapping and extents
     *
     One accessor consists substantially of an array of offsets (runtime values), a extent and an index (copmpile-time constants). The latter is used to distinguish the types of two different accessors,
     while the offsets are used to calculate the final memory address to be accessed by the stencil in \ref gridtools::iterate_domain.
     * The class also provides the interface for accessing data in the function body.
     The interfaces available to specify the offset in each dimension are covered in the following example, supposing that we have to specify the offsets of a 3D field V:
     - specify three offsets: in this case the order matters, the three arguments represent the offset in the  i, j, k directions respectively.
     \verbatim
     V(1,0,-3)
     \endverbatim
     - specify only some components: in this case the order is arbitrary and the missing components have offset zero;
     \verbatim
     V(z(-3),x(1))
     \endverbatim
     *
     * @tparam I Index of the argument in the function argument list
     * @tparam Extend Bounds over which the function access the argument
     */
    template <uint_t I, enumtype::intend Intend, typename Extend, ushort_t Dim >
    struct accessor_base  {

        //typedef useful when unnecessary indirections are used
        typedef accessor_base<I, Intend, Extend, Dim> type;
        template <uint_t II, enumtype::intend It, typename R, ushort_t D>
        friend std::ostream& operator<<(std::ostream& s, accessor_base<II,It,R,D> const& x);

        typedef accessor_base<I, Intend, Extend, Dim> base_t;
        static const ushort_t n_dim=Dim;

        typedef static_uint<I> index_type;
        typedef enumtype::enum_type<enumtype::intend, Intend> intend_t;
        typedef Extend extent_type;

    private:
        offset_tuple<n_dim, n_dim> m_offsets;

    public:
        /**@brief Default constructor
           NOTE: the following constructor when used with the brace initializer produces with nvcc a considerable amount of extra instructions (gcc 4.8.2), and degrades the performances (which is probably a compiler bug, I couldn't reproduce it on a small test).*/
        GT_FUNCTION
        constexpr explicit accessor_base():m_offsets() {}

#if defined( CXX11_ENABLED ) && ! defined(__CUDACC__)
        //move ctor
        GT_FUNCTION
        constexpr accessor_base(const type && other) : m_offsets(other.m_offsets){}

        //move ctor from another accessor_base with different index
        template<uint_t OtherIndex>
        GT_FUNCTION
        constexpr accessor_base(accessor_base<OtherIndex, Intend, Extend, Dim>&& other) :
            m_offsets(other.offsets()) {}
#endif
        //copy ctor
        GT_FUNCTION
        constexpr accessor_base( type const& other) : m_offsets(other.m_offsets){}

        //copy ctor from another accessor_base with different index
        template<uint_t OtherIndex>
        GT_FUNCTION
        constexpr accessor_base(const accessor_base<OtherIndex, Intend, Extend, Dim> & other) :
            m_offsets(other.offsets()){}

        //ctor with one argument have to provide specific arguments in order to avoid ambiguous instantiation
        // by the compiler
        template<uint_t Idx>
        GT_FUNCTION
        constexpr accessor_base (dimension<Idx> const& x ): m_offsets(x) {
            GRIDTOOLS_STATIC_ASSERT((Idx <= n_dim), "too high dimension accessor");
        }

        GT_FUNCTION
        constexpr accessor_base (const int_t x ): m_offsets(x) {}


        /**@brief constructor taking the dimension class as argument.
           This allows to specify the extra arguments out of order. Note that 'dimension' is a
           language keyword used at the interface level.
        */
#if defined( CXX11_ENABLED ) && ! defined(__CUDACC__) //cuda messing up
        template <typename... Whatever, typename Dummy=all_integers<Whatever ...> >
        GT_FUNCTION
        constexpr accessor_base ( Whatever... x) : m_offsets( x...)
        {
            GRIDTOOLS_STATIC_ASSERT(sizeof...(x)<=n_dim, "the number of arguments passed to the offset_tuple constructor exceeds the number of space dimensions of the storage. Check that you are not accessing a non existing dimension, or increase the dimension D of the accessor (accessor<Id, extent, D>)");
        }

        template <ushort_t ... Idx >
        GT_FUNCTION
        constexpr accessor_base ( dimension<Idx>... x) : m_offsets( x...)
        {
            GRIDTOOLS_STATIC_ASSERT(accumulate(logical_and(), (Idx <= n_dim) ...), "trying to access a too high dimension for accessor");
            GRIDTOOLS_STATIC_ASSERT(sizeof...(x)<=n_dim, "the number of arguments passed to the offset_tuple constructor exceeds the number of space dimensions of the storage. Check that you are not accessing a non existing dimension, or increase the dimension D of the accessor (accessor<Id, extent, D>)");
        }
#else

        template<typename X, typename Y, typename Z, typename T, typename U, typename V>
        GT_FUNCTION
        constexpr accessor_base (X x, Y y, Z z, T t, U u, V v ): m_offsets(x,y,z,y,u,v)
        {}

        template<typename X, typename Y, typename Z, typename T, typename U>
        GT_FUNCTION
        constexpr accessor_base (X x, Y y, Z z, T t, U u ): m_offsets(x,y,z,y, u)
        {}

        template<typename X, typename Y, typename Z, typename T>
        GT_FUNCTION
        constexpr accessor_base (X x, Y y, Z z, T t ): m_offsets(x,y,z,y)
        {}

        template<typename X, typename Y, typename Z>
        GT_FUNCTION
        constexpr accessor_base (X x, Y y, Z z ): m_offsets(x,y,z)
        {}

        template<typename X, typename Y>
        GT_FUNCTION
        constexpr accessor_base (X x, Y y ): m_offsets(x,y) {}
#endif

        static  void info() {
            std::cout << "Arg_type storage with index " << I << " and extent " << Extend() << " ";
        }

        template<short_t Idx>
        GT_FUNCTION
        constexpr bool end() const {return true;}

        template<short_t Idx>
        GT_FUNCTION
        constexpr
        int_t get() const {
            GRIDTOOLS_STATIC_ASSERT(Idx>=0, "requested accessor index lower than zero. Check that when you define the accessor you specify the dimenisons which you actually access. e.g. suppose that a storage linked to the accessor ```in``` has 5 dimensions, and thus can be called with in(Dimensions<5>(-1)). Calling in(Dimensions<6>(-1)) brings you here.");
            GRIDTOOLS_STATIC_ASSERT(Idx<=n_dim, "requested accessor index larger than the available dimensions. Maybe you made a mistake when setting the accessor dimensionality?");
            return m_offsets.template get<Idx>();
        }

        GT_FUNCTION
        constexpr const offset_tuple<n_dim, n_dim>& offsets() const { return m_offsets;}

    };

//################################################################################
//                              Compile time checks
//################################################################################

    /**
     * Struct to test if an argument (placeholder) is an argument (placeholder)
     */
    template <typename T>
    struct is_plchldr : boost::false_type
    {};

    /**
     * Struct to test if an argument is a placeholder - Specialization yielding true
     */
    template <uint_t I, typename T, typename C>
    struct is_plchldr<arg<I,T,C> > : boost::true_type
    {};

    /**
     * Struct to test if an argument (placeholder) is a temporary
     */
    template <typename T>
    struct is_plchldr_to_temp : boost::mpl::false_{};

    /**
     * Struct to test if an argument (placeholder) is a temporary no_storage_type_yet - Specialization yielding true
     */
    template <uint_t I, typename T, typename C>
    struct is_plchldr_to_temp<arg<I, no_storage_type_yet<T>, C > > : boost::true_type
    {};

    /**
     * Struct to test if an argument is a placeholder to a temporary storage
     */
    template <uint_t I, typename T, typename U, ushort_t Dim, typename C>
    struct is_plchldr_to_temp<arg<I, base_storage< T, U, Dim>, C > > : boost::mpl::bool_<U::is_temporary>
    {};

    /**
     * Struct to test if an argument is a temporary
     no_storage_type_yet - Specialization for a decorator of the
     storage class, falls back on the original class type here the
     decorator is the \ref gridtools::storage
    */
    template <uint_t I, typename BaseType, template <typename T> class Decorator, typename C>
    struct is_plchldr_to_temp<arg<I, Decorator<BaseType>, C > > : is_plchldr_to_temp<arg<I, typename BaseType::basic_type, C> >
    {};

    template <uint_t I, typename BaseType, typename C>
    struct is_plchldr_to_temp<arg<I, storage<BaseType>, C > > : is_plchldr_to_temp<arg<I, typename BaseType::basic_type, C> >
    {};

    /**
     * Printing type information for debug purposes
     * @param s The ostream
     * @param n/a Type selector for offset_tuple
     * @return ostream
     */
    template <uint_t I, enumtype::intend It, typename R, ushort_t D>
    std::ostream& operator<<(std::ostream& s, accessor_base<I,It,R,D> const& x) {
        s << "[ offset_tuple< " << I
                 << ", " << R()
                 << ", " << It
                 << ", " << D
                 // << " (" << x.i()
                 // << ", " << x.j()
                 // << ", " << x.k()
                 <<" ) > m_offset: {";

        for (int i=0; i<x.n_dim-1; ++i) {
            s << x.m_offset[i] << ", ";
        }
        s << x.m_offset[x.n_dim-1] << "} ]";
        return s;
    }

    /**
     * Printing type information for debug purposes
     * @param s The ostream
     * @param n/a Type selector for offset_tuple
     * @return ostream
     */
    template <uint_t I, typename R, typename C>
    std::ostream& operator<<(std::ostream& s, arg<I,no_storage_type_yet<R>, C > const&) {
        return s << "[ arg< " << I
                 << ", temporary<something>" << " > ]";
    }

    /**
     * Printing type information for debug purposes
<     * @param s The ostream
     * @param n/a Type selector for arg to a NON temp
     * @return ostream
     */
    template <uint_t I, typename R, typename C>
    std::ostream& operator<<(std::ostream& s, arg<I,R,C> const&) {
        return s << "[ arg< " << I
                 << ", NON TEMP" << " > ]";
    }
} // namespace gridtools
