#pragma once
#include "./arg.hpp"

namespace gridtools {

    namespace impl {
        /** metafunction that associates (in a mpl::map) placeholders to extents.
        * It returns a mpl::map between placeholders and extents of the local arguments.
        */
        template < typename Placeholders, typename Accessors >
        struct make_arg_with_extent_map {

            GRIDTOOLS_STATIC_ASSERT((is_sequence_of< Placeholders, is_arg >::value), "Internal Error: wrong type");
            GRIDTOOLS_STATIC_ASSERT(
                (is_sequence_of< Accessors, is_any_accessor >::value), "Internal Error: wrong type");
#ifdef PEDANTIC // with global accessors this assertion fails (since they are not in the Accessors)
            GRIDTOOLS_STATIC_ASSERT((boost::mpl::size< Placeholders >::value == boost::mpl::size< Accessors >::value),
                "Size of placeholder arguments passed to esf \n"
                "    make_esf<functor>(arg1(), arg2()) )\n"
                "does not match the list of arguments defined within the ESF, like\n"
                "    typedef boost::mpl::vector<arg_in, arg_out> arg_list.");
#endif
            template < typename Accessor >
            struct _get_extent {
                typedef typename Accessor::extent_t type;
            };

            /** Given the list of placeholders (Plcs) and the list of arguemnts of a
                stencil operator (Accessors), this struct will insert the placeholder type
                (as key) and the corresponding extent into an mpl::map.
            */
            template < typename Plcs, typename LArgs >
            struct from {
                template < typename CurrentMap, typename Index >
                struct insert {

                    typedef typename boost::mpl::at_c< LArgs, Index::value >::type accessor_t;
                    typedef typename boost::mpl::insert< CurrentMap,
                        typename boost::mpl::pair< typename boost::mpl::at_c< Plcs, Index::value >::type,
                                                             typename _get_extent< accessor_t >::type > >::type type;
                };
            };

            // Note: only the accessors of storage type are considered in the sequence
            typedef typename boost::mpl::range_c< uint_t, 0, boost::mpl::size< Accessors >::type::value > iter_range;

            /** Here the iteration begins by filling an empty map */
            typedef typename boost::mpl::fold< iter_range,
                boost::mpl::map0<>,
                typename from< Placeholders, Accessors >::template insert< boost::mpl::_1, boost::mpl::_2 > >::type
                type;
        };
    }
}
