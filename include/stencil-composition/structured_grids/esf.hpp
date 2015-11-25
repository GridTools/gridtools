#pragma once

#include <boost/type_traits/is_const.hpp>

#include "accessor.hpp"
#include "stencil-composition/domain_type.hpp"
#include "common/generic_metafunctions/is_sequence_of.hpp"
#include "stencil-composition/esf_fwd.hpp"
#include "stencil-composition/sfinae.hpp"

/**
   @file
   @brief Descriptors for Elementary Stencil Function (ESF)
*/
namespace gridtools {

    /**
     * @brief Descriptors for Elementary Stencil Function (ESF)
     */
    template <typename ESF, typename ArgArray, typename Staggering=staggered<0,0,0,0,0,0> >
    struct esf_descriptor {
        GRIDTOOLS_STATIC_ASSERT((is_sequence_of<ArgArray, is_arg>::value), "wrong types for the list of parameter placeholders\n"
                "check the make_esf syntax");
    private:

        /** Private metafunction that associates (in a mpl::map) placeholders to extents.
            It returns a mpl::map between placeholders and extents of the local arguments.
         */
        template <typename Placeholders, typename LocalArgs>
        struct _make_map {

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

            typedef typename boost::mpl::range_c<uint_t, 0, boost::mpl::size<Placeholders>::type::value> iter_range;

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
        GRIDTOOLS_STATIC_ASSERT(has_arg_list<esf_function>::type::value, "The type arg_list was not found in a user functor definition. All user functors must have a type alias called \'arg_list\', which is an MPL vector containing the list of accessors defined in the functor (NOTE: the \'generic_accessor\' types are excluded from this list). Example: \n\n using v1=accessor<0>; \n using v2=generic_accessor<1>; \n using v3=accessor<2>; \n using arg_list=boost::mpl::vector<v1, v3>;");
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
                                "You specified different placeholders with the same index. Check the indexes of the arg_type definitions.");

        //checking if the index list contains holes (a common error is to define a list of types with indexes which are not contiguous)
        typedef typename boost::mpl::find_if<raw_index_list, boost::mpl::greater<boost::mpl::_1, static_int<len-1> > >::type test;
        //check if the index list contains holes (a common error is to define a list of types with indexes which are not contiguous)
        GRIDTOOLS_STATIC_ASSERT((boost::is_same<typename test::type, boost::mpl::void_ >::value) , "the index list contains holes:\n "
                                "The numeration of the placeholders is not contiguous. You have to define each arg_type with a unique identifier ranging "
                                " from 1 to N without \"holes\".");
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
