#pragma once

#include "intermediate.h"

namespace gridtools {
    template<typename T> struct intermediate_backend;

    template <typename Backend,
              typename LayoutType,
              typename MssArray,
              typename DomainType,
              typename Coords,
              bool IsStateful>
    struct intermediate_backend<intermediate<Backend, LayoutType, MssArray, DomainType, Coords, IsStateful> >
    {
        typedef Backend type;
    };

    template<typename T> struct intermediate_domain_type;

    template <typename Backend,
              typename LayoutType,
              typename MssArray,
              typename DomainType,
              typename Coords,
              bool IsStateful>
    struct intermediate_domain_type<intermediate<Backend, LayoutType, MssArray, DomainType, Coords, IsStateful> >
    {
        typedef DomainType type;
    };

    template<typename T> struct intermediate_mss_array;

    template <typename Backend,
              typename LayoutType,
              typename MssArray,
              typename DomainType,
              typename Coords,
              bool IsStateful>
    struct intermediate_mss_array<intermediate<Backend, LayoutType, MssArray, DomainType, Coords, IsStateful> >
    {
        typedef MssArray type;
    };

    template<typename T> struct intermediate_layout_type;

    template <typename Backend,
              typename LayoutType,
              typename MssArray,
              typename DomainType,
              typename Coords,
              bool IsStateful>
    struct intermediate_layout_type<intermediate<Backend, LayoutType, MssArray, DomainType, Coords, IsStateful> >
    {
        typedef LayoutType type;
    };

    template<typename T> struct intermediate_is_stateful;

    template <typename Backend,
              typename LayoutType,
              typename MssArray,
              typename DomainType,
              typename Coords,
              bool IsStateful>
    struct intermediate_is_stateful<intermediate<Backend, LayoutType, MssArray, DomainType, Coords, IsStateful> >
    {
        typedef boost::mpl::bool_<IsStateful> type;
    };
}//namespace gridtools
