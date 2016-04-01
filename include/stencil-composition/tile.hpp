#pragma once
namespace gridtools {

    template < uint_t Tile, uint_t Minus, uint_t Plus >
    struct tile {
        static const uint_t s_minus = Minus;
        static const uint_t s_plus = Plus;
        static const uint_t s_tile = Tile;
        typedef static_uint< s_minus > s_minus_t;
        typedef static_uint< s_plus > s_plus_t;
        typedef static_uint< s_tile > s_tile_t;
    };

    template < typename T >
    struct is_tile : boost::mpl::false_ {};

    template < uint_t Tile, uint_t Minus, uint_t Plus >
    struct is_tile< tile< Tile, Minus, Plus > > : boost::mpl::true_ {};
}
