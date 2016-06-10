#pragma once
#include "storage/partitioner.hpp"
#include "stencil_composition/axis.hpp"

namespace gridtools {

    template < typename Axis, typename Partitioner = partitioner_dummy >
    struct grid : public clonable_to_gpu< grid< Axis, Partitioner > > {
        GRIDTOOLS_STATIC_ASSERT((is_interval< Axis >::value), "Internal Error: wrong type");
        typedef Axis axis_type;
        typedef Partitioner partitioner_t;

        typedef typename boost::mpl::plus<
            boost::mpl::minus< typename Axis::ToLevel::Splitter, typename Axis::FromLevel::Splitter >,
            static_int< 1 > >::type size_type;

        array< uint_t, size_type::value > value_list;

        GT_FUNCTION
        explicit grid(halo_descriptor const &direction_i, halo_descriptor const &direction_j)
            : m_partitioner(partitioner_dummy()), m_direction_i(direction_i), m_direction_j(direction_j) {
            GRIDTOOLS_STATIC_ASSERT(is_partitioner_dummy< partitioner_t >::value,
                "you have to construct the grid with a valid partitioner, or with no partitioner at all.");
        }

        template < typename ParallelStorage >
        GT_FUNCTION explicit grid(const Partitioner &part_, ParallelStorage const &storage_)
            : m_partitioner(part_), m_direction_i(storage_.template get_halo_descriptor< 0 >()) // copy
              ,
              m_direction_j(storage_.template get_halo_descriptor< 1 >()) // copy
        {
            GRIDTOOLS_STATIC_ASSERT(!is_partitioner_dummy< Partitioner >::value,
                "you have to add the partitioner to the grid template parameters");
        }

        GT_FUNCTION
        explicit grid(uint_t *i, uint_t *j /*, uint_t* k*/)
            : m_partitioner(partitioner_dummy()) // ok since partitioner_dummy is empty. Generates a warning
              ,
              m_direction_i(i[minus], i[plus], i[begin], i[end], i[length]),
              m_direction_j(j[minus], j[plus], j[begin], j[end], j[length]) {
            GRIDTOOLS_STATIC_ASSERT(is_partitioner_dummy< partitioner_t >::value,
                "you have to construct the grid with a valid partitioner, or with no partitioner at all.");
        }

        GT_FUNCTION
        explicit grid(array< uint_t, 5 > &i, array< uint_t, 5 > &j)
            : m_partitioner(partitioner_dummy()) // ok since partitioner_dummy is empty. Generates a warning
              ,
              m_direction_i(i[minus], i[plus], i[begin], i[end], i[length]),
              m_direction_j(j[minus], j[plus], j[begin], j[end], j[length]) {
            GRIDTOOLS_STATIC_ASSERT(is_partitioner_dummy< partitioner_t >::value,
                "you have to construct the grid with a valid partitioner, or with no partitioner at all.");
        }

        GT_FUNCTION
        uint_t i_low_bound() const { return m_direction_i.begin(); }

        GT_FUNCTION
        uint_t i_high_bound() const { return m_direction_i.end(); }

        GT_FUNCTION
        uint_t j_low_bound() const { return m_direction_j.begin(); }

        GT_FUNCTION
        uint_t j_high_bound() const { return m_direction_j.end(); }

        template < typename Level >
        GT_FUNCTION uint_t value_at() const {
            GRIDTOOLS_STATIC_ASSERT((is_level< Level >::value), "Internal Error: wrong type");
            int_t offs = Level::Offset::value;
            if (offs < 0)
                offs += 1;
            return value_list[Level::Splitter::value] + offs;
        }

        GT_FUNCTION
        uint_t value_at_top() const {
            return value_list[size_type::value - 1];
            // return m_k_high_bound;
        }

        GT_FUNCTION
        uint_t value_at_bottom() const {
            return value_list[0];
            // return m_k_low_bound;
        }

        halo_descriptor const &direction_i() const { return m_direction_i; }

        halo_descriptor const &direction_j() const { return m_direction_j; }

        const Partitioner &partitioner() const {
            // the partitioner must be set
            return m_partitioner;
        }

        template < typename Flag >
        bool at_boundary(ushort_t const &coordinate_, Flag const &flag_) const {
            return m_partitioner.at_boundary(coordinate_, flag_);
        }

      private:
        Partitioner const &m_partitioner;
        halo_descriptor m_direction_i;
        halo_descriptor m_direction_j;
    };

    template < typename Grid >
    struct is_grid : boost::mpl::false_ {};

    template < typename Axis >
    struct is_grid< grid< Axis > > : boost::mpl::true_ {};

    template < typename Axis, typename Partitioner >
    struct is_grid< grid< Axis, Partitioner > > : boost::mpl::true_ {};
}
