#pragma once
#include "direction.hpp"

namespace gridtools {

    struct default_predicate {
        template <typename Direction>
        bool operator()(Direction) const {
            return true;
        }
    };

    struct bitmap_predicate {
        uint_t m_boundary_bitmap; // see storage/partitioner_trivial.hpp

        enum Flag{UP=1, LOW=8};

        bitmap_predicate(uint_t bm)
            : m_boundary_bitmap(bm)
        {}

        template <sign I, sign J, sign K>
        bool operator()(direction<I,J,K>) const {
            return true;
            // return (at_boundary(0, ((I==minus_)?UP:LOW) )) ||
            //     (at_boundary(1, ((J==minus_)?UP:LOW))) ||
            //     (at_boundary(2, ((K==minus_)?UP:LOW)));
        }

    private:
        GT_FUNCTION
        bool at_boundary(ushort_t const& component_, Flag flag_) const {
            return !(
                m_boundary_bitmap%(ushort_t)((ushort_t)gt_pow<2>::apply(component_+1)*(ushort_t)flag_)
                <
                ((component_+(ushort_t)1)*(ushort_t)flag_)
                );
        }

    };

} // namespace gridtools
