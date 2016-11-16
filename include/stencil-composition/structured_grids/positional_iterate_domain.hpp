#include "iterate_domain.hpp"

namespace gridtools{
    /**@brief class handling the computation of the */
    template < typename IterateDomainImpl >
    struct positional_iterate_domain : public iterate_domain< IterateDomainImpl > {
        typedef iterate_domain< IterateDomainImpl > base_t;
        typedef typename base_t::reduction_type_t reduction_type_t;
        typedef typename base_t::local_domain_t local_domain_t;

        using iterate_domain< IterateDomainImpl >::iterate_domain;

        /**@brief method for incrementing the index when moving forward along the k direction */
        template < ushort_t Coordinate, typename Execution >
        GT_FUNCTION void increment() {
            if (Coordinate == 0) {
                m_i += Execution::value;
            }
            if (Coordinate == 1) {
                m_j += Execution::value;
            }
            if (Coordinate == 2)
                m_k += Execution::value;
            base_t::template increment< Coordinate, Execution >();
        }

        /**@brief method for incrementing the index when moving forward along the k direction */
        template < ushort_t Coordinate >
        GT_FUNCTION void increment(const uint_t steps_) {
            if (Coordinate == 0) {
                m_i += steps_;
            }
            if (Coordinate == 1) {
                m_j += steps_;
            }
            if (Coordinate == 2)
                m_k += steps_;
            base_t::template increment< Coordinate >(steps_);
        }

        template < ushort_t Coordinate >
        GT_FUNCTION void initialize(uint_t const &index = 0, uint_t const &block = 0) {
            if (Coordinate == 0) {
                m_i = index;
            }
            if (Coordinate == 1) {
                m_j = index;
            }
            if (Coordinate == 2) {
                m_k = index;
            }
            base_t::template initialize< Coordinate >(index, block);
        }

        template < ushort_t Coordinate >
        GT_FUNCTION void reset_positional_index(uint_t const &lowerbound = 0) {
            if (Coordinate == 0) {
                m_i = lowerbound;
            }
            if (Coordinate == 1) {
                m_j = lowerbound;
            }
            if (Coordinate == 2) {
                m_k = lowerbound;
            }
        }

        GT_FUNCTION
        uint_t i() const { return m_i; }

        GT_FUNCTION
        uint_t j() const { return m_j; }

        GT_FUNCTION
        uint_t k() const { return m_k; }

      private:
        uint_t m_i, m_j, m_k;
    };
}
