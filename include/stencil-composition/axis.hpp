#pragma once

#include <boost/mpl/minus.hpp>
#include <boost/mpl/plus.hpp>
#include "loopintervals.hpp"
#include "../common/halo_descriptor.hpp"
#include "../common/gpu_clone.hpp"
#include "storage/partitioner.hpp"
/**@file
@brief file containing the size of the horizontal domain

The domain is currently described in terms of 2 horiozntal axis of type \ref gridtools::halo_descriptor , and the vertical axis bounds which are treated separately.
TODO This should be easily generalizable to arbitrary dimensions
*/
namespace gridtools {
    template <typename MinLevel, typename MaxLevel>
    struct make_axis {
        typedef interval<MinLevel, MaxLevel> type;
    };

    template <typename Axis, uint_t I>
    struct extend_by {
        typedef interval<level<Axis::FromLevel::Splitter::value, Axis::FromLevel::Offset::value - 1>,
                         level<Axis::ToLevel::Splitter::value, Axis::ToLevel::Offset::value + 1> > type;
    };

    namespace    enumtype_axis{
        enum coordinate_argument {minus, plus, begin, end, length};
    }//namespace enumtype_axis

    using namespace enumtype_axis;


    template <typename Axis, typename Partitioner=partitioner_dummy>
    struct coordinates : public clonable_to_gpu<coordinates<Axis, Partitioner> > {
        GRIDTOOLS_STATIC_ASSERT((is_interval<Axis>::value), "Internal Error: wrong type");
        typedef Axis axis_type;
        typedef Partitioner partitioner_t;

        typedef typename boost::mpl::plus<
                boost::mpl::minus<typename Axis::ToLevel::Splitter, typename Axis::FromLevel::Splitter>,
                static_int<1>
        >::type size_type;

        array<uint_t, size_type::value > value_list;

        GT_FUNCTION
        explicit coordinates( halo_descriptor const& direction_i, halo_descriptor const& direction_j):
            m_partitioner(partitioner_dummy()),
            m_direction_i(direction_i),
            m_direction_j(direction_j)
        {
            GRIDTOOLS_STATIC_ASSERT(is_partitioner_dummy<partitioner_t>::value, "you have to construct the coordinates with a valid partitioner, or with no partitioner at all.");
        }


        template<typename ParallelStorage>
        GT_FUNCTION
        explicit coordinates( const Partitioner& part_, ParallelStorage const & storage_ )
            :
            m_partitioner(part_)
            , m_direction_i(storage_.template get_halo_descriptor<0>())//copy
            , m_direction_j(storage_.template get_halo_descriptor<1>())//copy
        {
            GRIDTOOLS_STATIC_ASSERT(!is_partitioner_dummy<Partitioner>::value,
                                    "you have to add the partitioner to the coordinates template parameters");
        }

        GT_FUNCTION
        explicit coordinates( uint_t* i, uint_t* j/*, uint_t* k*/)
            :
            m_partitioner(partitioner_dummy())//ok since partitioner_dummy is empty. Generates a warning
            , m_direction_i(i[minus], i[plus], i[begin], i[end], i[length])
            , m_direction_j(j[minus], j[plus], j[begin], j[end], j[length])
        {
            GRIDTOOLS_STATIC_ASSERT(is_partitioner_dummy<partitioner_t>::value, "you have to construct the coordinates with a valid partitioner, or with no partitioner at all.");
        }

        GT_FUNCTION
        explicit coordinates( array<uint_t, 5>& i, array<uint_t, 5>& j)
            :
            m_partitioner(partitioner_dummy())//ok since partitioner_dummy is empty. Generates a warning
            , m_direction_i(i[minus], i[plus], i[begin], i[end], i[length])
            , m_direction_j(j[minus], j[plus], j[begin], j[end], j[length])
        {
            GRIDTOOLS_STATIC_ASSERT(is_partitioner_dummy<partitioner_t>::value, "you have to construct the coordinates with a valid partitioner, or with no partitioner at all.");
        }


        GT_FUNCTION
        uint_t i_low_bound() const {
            return m_direction_i.begin();
        }

        GT_FUNCTION
        uint_t i_high_bound() const {
            return m_direction_i.end();
        }

        GT_FUNCTION
        uint_t j_low_bound() const {
            return m_direction_j.begin();
        }

        GT_FUNCTION
        uint_t j_high_bound() const {
            return m_direction_j.end();
        }

        template <typename Level>
        GT_FUNCTION
        uint_t value_at() const {
            GRIDTOOLS_STATIC_ASSERT((is_level<Level>::value), "Internal Error: wrong type");
            int_t offs = Level::Offset::value;
            if (offs < 0) offs += 1;
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

        halo_descriptor const& direction_i() const { return m_direction_i;}

        halo_descriptor const& direction_j() const { return m_direction_j;}

        const Partitioner & partitioner() const {
            //the partitioner must be set
            return m_partitioner;}

        template<typename Flag>
        bool at_boundary(ushort_t const& coordinate_, Flag const& flag_) const
        {
            return m_partitioner.at_boundary(coordinate_, flag_);
        }

    private:

        Partitioner const& m_partitioner;
        halo_descriptor m_direction_i;
        halo_descriptor m_direction_j;

    };

    template<typename Coord>
    struct is_coordinates : boost::mpl::false_{};

    template<typename Axis>
    struct is_coordinates<coordinates<Axis> > : boost::mpl::true_{};

    template<typename Axis, typename Partitioner>
    struct is_coordinates<coordinates<Axis, Partitioner> > : boost::mpl::true_{};

} // namespace gridtools
