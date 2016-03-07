#pragma once

#include <boost/type_traits/is_const.hpp>

#include "accessor.hpp"
#include "../domain_type.hpp"
#include "../../common/generic_metafunctions/is_sequence_of.hpp"
#include "../esf_fwd.hpp"
#include "../sfinae.hpp"

/**
   @file
   @brief Descriptors for Elementary Stencil Function (ESF)
*/
namespace gridtools {

    namespace _impl {
        /**
           Metafunction to check that the arg_list mpl::vector list the
           different accessors in order!
        */
        template <typename ArgList>
        struct check_arg_list {
            template <typename Reduced, typename Element>
            struct _check {
                typedef typename boost::mpl::if_c<
                    (Element::index_type::value == Reduced::value+1),
                    boost::mpl::int_<Reduced::value+1>,
                    boost::mpl::int_<-Reduced::value-1>
                    >::type type;
            };

            typedef typename boost::mpl::fold<
                ArgList,
                boost::mpl::int_<-1>,
                _check<boost::mpl::_1, boost::mpl::_2>
                >::type res_type;

            typedef typename boost::mpl::if_c<
                (res_type::value+1 == boost::mpl::size<ArgList>::value),
                boost::true_type,
                boost::false_type>::type type;

            static const bool value = type::value;
        };
    } // namespace _impl

    /**
     * @brief Descriptors for Elementary Stencil Function (ESF)
     */
    template <typename ESF, typename ArgArray, typename Staggering=staggered<0,0,0,0,0,0> >
    struct esf_descriptor {
        GRIDTOOLS_STATIC_ASSERT((is_sequence_of<ArgArray, is_arg>::value), "wrong types for the list of parameter placeholders\n"
                "check the make_esf syntax, it seems you passed in a placeholder with the wrong type (it should be of type gridtools::arg ).");
    private:

        GRIDTOOLS_STATIC_ASSERT((_impl::check_arg_list<typename ESF::arg_list>::value == true), "There is a problem in the arg_list a the functor. Please, ensure that the accessors are listed in order from index 0 to the last");

        /** Private metafunction that associates (in a mpl::map) placeholders to extents.
            It returns a mpl::map between placeholders and extents of the local arguments.
         */
        template <typename Placeholders, typename LocalArgs>
        struct _make_map {

#ifdef PEDANTIC //with global accessors this assertion fails (since they are not in the LocalArgs)
            GRIDTOOLS_STATIC_ASSERT((boost::mpl::size<Placeholders>::value == boost::mpl::size<LocalArgs>::value),
                 "Size of placeholder arguments passed to esf \n"
                 "    make_esf<functor>(arg1(), arg2()) )\n"
                 "does not match the list of arguments defined within the ESF, like\n"
                 "    typedef boost::mpl::vector<arg_in, arg_out> arg_list."
                 );
#endif
            /** Given the list of placeholders (Plcs) and the list of arguemnts of a
                stencil operator (LocalArgs), this struct will insert the placeholder type
                (as key) and the corresponding extent into an mpl::map.
             */
            template <typename Plcs, typename LArgs>
            struct from {
                template <typename CurrentMap, typename Index>
                struct insert {
                    typedef typename boost::mpl::insert<
                        CurrentMap,
                        typename boost::mpl::pair<
                            typename boost::mpl::at_c<Plcs, Index::value>::type,
                            typename boost::mpl::at_c<LArgs, Index::value>::type::extent_type
                            >
                        >::type type;
                };
            };

            //Note: only the accessors of storage type are considered in the sequence
            typedef typename boost::mpl::range_c<uint_t, 0, boost::mpl::size<LocalArgs>::type::value> iter_range;

            /** Here the iteration begins by filling an empty map */
            typedef typename boost::mpl::fold<
                iter_range,
                boost::mpl::map0<>,
                typename from<Placeholders, LocalArgs>::template insert<boost::mpl::_1, boost::mpl::_2>
                >::type type;
        };

    public:
        typedef ESF esf_function;
        typedef ArgArray args_t;

        /** Type member with the mapping between placeholder types (as key) to extents in the operator */
        typedef typename _make_map<args_t, typename esf_function::arg_list>::type args_with_extents;
        typedef Staggering staggering_t;

        //////////////////////Compile time checks ////////////////////////////////////////////////////////////

        /**@brief Macro defining a sfinae metafunction

           defines a metafunction has_extent_type, which returns true if its template argument
           defines a type called arg_list. It also defines a get_arg_list metafunction, which
           can be used to return the arg_list only when it is present, without giving compilation
           errors in case it is not defined.
        */
        HAS_TYPE_SFINAE(arg_list, has_arg_list, get_arg_list)
        GRIDTOOLS_STATIC_ASSERT(has_arg_list<esf_function>::type::value, "The type arg_list was not found in a user functor definition. All user functors must have a type alias called \'arg_list\', which is an MPL vector containing the list of accessors defined in the functor (NOTE: the \'global_accessor\' types are excluded from this list). Example: \n\n using v1=accessor<0>; \n using v2=generic_accessor<1, enumtype::in>; \n using v3=accessor<2>; \n using arg_list=boost::mpl::vector<v1, v3>;");
        //checking that all the placeholders have a different index
        /**
         * \brief Get a sequence of the same type as original_placeholders, containing the indexes relative to the placehoolders
         * note that the static const indexes are transformed into types using mpl::integral_c
         */
        typedef _impl::compute_index_set<typename esf_function::arg_list> check_holes;
        typedef typename check_holes::raw_index_list raw_index_list;
        typedef typename check_holes::index_set index_set;
        static const ushort_t len=check_holes::len;

        //actual check if the user specified placeholder arguments with the same index
        GRIDTOOLS_STATIC_ASSERT((len == boost::mpl::size<index_set>::type::value ),
                                "You specified different accessors with the same index. Check the indexes of the accessor definitions in the functor.");

        //checking if the index list contains holes (a common error is to define a list of types with indexes which are not contiguous)
        typedef typename boost::mpl::find_if<raw_index_list, boost::mpl::greater<boost::mpl::_1, static_int<len-1> > >::type test;
        //check if the index list contains holes (a common error is to define a list of types with indexes which are not contiguous)
        GRIDTOOLS_STATIC_ASSERT((boost::is_same<typename test::type, boost::mpl::void_ >::value) , "the index list contains holes:\n "
                                "The numeration of the placeholders is not contiguous. You have to define each accessor with a unique identifier ranging "
                                " from 0 to N-1 without \"holes\". Check also that the mpl vector arg_type containd all the accessors used.");
        //////////////////////////////////////////////////////////////////////////////////////////////////////
    };

    template <typename T, typename V>
    std::ostream& operator<<(std::ostream& s, esf_descriptor<T,V> const7) {
        return s << "esf_desctiptor< " << T()
                 << " with " << boost::mpl::size<typename esf_descriptor<T,V>::args_t>::type::value
                 << " arguments (double check "
                 << boost::mpl::size<typename esf_descriptor<T,V>::esf_function::arg_list>::type::value << ")";
    }

    template<typename ESF, typename ArgArray, typename Staggering>
    struct is_esf_descriptor<esf_descriptor<ESF, ArgArray, Staggering> > : boost::mpl::true_{};

} // namespace gridtools
