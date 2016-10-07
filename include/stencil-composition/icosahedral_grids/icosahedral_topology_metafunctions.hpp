#pragma once
#include "../../common/defs.hpp"
#include "../../common/gt_math.hpp"
#include "../../common/selector.hpp"
#include "../../common/array.hpp"
#include "../location_type.hpp"
#include "../../common/generic_metafunctions/pack_get_elem.hpp"
#include "../../common/generic_metafunctions/gt_integer_sequence.hpp"

namespace gridtools {
    namespace impl {
        template < uint_t Pos >
        constexpr long long compute_uuid_selector() {
            return 0;
        }

        template < uint_t Pos, typename... Int >
        constexpr long long compute_uuid_selector(int val0, Int... val) {
            return (val0 == 1) ? gt_pow< Pos >::apply((long long)2) + compute_uuid_selector< Pos + 1 >(val...)
                               : compute_uuid_selector< Pos + 1 >(val...);
        }

        template < int_t LocationTypeIndex, typename Selector >
        struct compute_uuid {};

        template < int_t LocationTypeIndex, int_t... Int >
        struct compute_uuid< LocationTypeIndex, selector< Int... > > {
            static constexpr ushort_t value =
                enumtype::metastorage_library_indices_limit + LocationTypeIndex + compute_uuid_selector< 2 >(Int...);
        };

        template < typename UInt, typename LocationType >
        struct array_elem_initializer {
            static_assert((is_location_type< LocationType >::value), "Error: expected a location type");

            template < int Idx >
            struct init_elem {
                GT_FUNCTION
                constexpr init_elem() {}

                GT_FUNCTION constexpr static UInt apply(const array< uint_t, 3 > space_dims) {
                    static_assert((Idx < 4), "Error");
                    return ((Idx == 0) ? space_dims[0]
                                       : ((Idx == 1) ? LocationType::n_colors::value : space_dims[Idx - 1]));
                }

                template < typename... ExtraInts >
                GT_FUNCTION constexpr static UInt apply(const array< uint_t, 3 > space_dims, ExtraInts... extra_dims) {
                    return ((Idx == 0)
                                ? space_dims[0]
                                : ((Idx == 1) ? LocationType::n_colors::value
                                              : (Idx < 4 ? space_dims[Idx - 1]
                                                         : pack_get_elem< Idx - 4 >::type::apply(extra_dims...))));
                }
            };
        };

        template < typename Uint, size_t ArraySize, typename LocationType, typename Selector >
        struct array_dim_initializers;

        template < typename UInt, size_t ArraySize, typename LocationType, int_t... Ints >
        struct array_dim_initializers< UInt, ArraySize, LocationType, selector< Ints... > > {
            static_assert((is_location_type< LocationType >::value), "Error: expected a location type");

            template < typename... ExtraInts >
            static constexpr array< UInt, ArraySize > apply(
                const array< uint_t, 3 > space_dims, ExtraInts... extra_dims) {
                using seq = apply_gt_integer_sequence< typename make_gt_integer_sequence< int, ArraySize >::type >;

                return seq::template apply< array< UInt, ArraySize >,
                    array_elem_initializer< UInt, LocationType >::template init_elem >(space_dims, extra_dims...);
            }
        };
    }
}
