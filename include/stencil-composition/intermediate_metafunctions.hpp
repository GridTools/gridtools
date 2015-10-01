#pragma once

#include "intermediate.hpp"

namespace gridtools {

    template<typename T> struct is_intermediate : boost::mpl::false_{};

    template <typename Backend,
              typename MssArray,
              typename DomainType,
              typename Coords,
              bool IsStateful>
    struct is_intermediate<intermediate<Backend, MssArray, DomainType, Coords, IsStateful> > :
        boost::mpl::true_{};

    template<typename T> struct intermediate_backend;

    template <typename Backend,
              typename MssArray,
              typename DomainType,
              typename Coords,
              bool IsStateful>
    struct intermediate_backend<intermediate<Backend, MssArray, DomainType, Coords, IsStateful> >
    {
        typedef Backend type;
    };

    template<typename T> struct intermediate_domain_type;

    template <typename Backend,
              typename MssArray,
              typename DomainType,
              typename Coords,
              bool IsStateful>
    struct intermediate_domain_type<intermediate<Backend, MssArray, DomainType, Coords, IsStateful> >
    {
        typedef DomainType type;
    };

    template<typename T> struct intermediate_mss_array;

    template <typename Backend,
              typename MssArray,
              typename DomainType,
              typename Coords,
              bool IsStateful>
    struct intermediate_mss_array<intermediate<Backend, MssArray, DomainType, Coords, IsStateful> >
    {
        typedef MssArray type;
    };

    template<typename Intermediate>
    struct intermediate_mss_components_array
    {
        GRIDTOOLS_STATIC_ASSERT((is_intermediate<Intermediate>::value), "Internal Error: wrong type");
        typedef typename Intermediate::mss_components_array_t type;
    };

    template<typename Intermediate>
    struct intermediate_range_sizes
    {
        GRIDTOOLS_STATIC_ASSERT((is_intermediate<Intermediate>::value), "Internal Error: wrong type");
        typedef typename Intermediate::range_sizes_t type;
    };

    template<typename T> struct intermediate_layout_type;

    template<typename T> struct intermediate_is_stateful;

    template <typename Backend,
              typename MssArray,
              typename DomainType,
              typename Coords,
              bool IsStateful>
    struct intermediate_is_stateful<intermediate<Backend, MssArray, DomainType, Coords, IsStateful> >
    {
        typedef boost::mpl::bool_<IsStateful> type;
    };

    template<typename T> struct intermediate_mss_local_domains;

    template<typename Intermediate>
    struct intermediate_mss_local_domains
    {
        GRIDTOOLS_STATIC_ASSERT((is_intermediate<Intermediate>::value), "Internal Error: wrong type");
        typedef typename Intermediate::mss_local_domains_t type;
    };

}//namespace gridtools
