#pragma once
#include <vector>

#include <vector>

namespace gridtools {

    template < typename MssDescriptorArray, bool HasReduction >
    struct reduction_data;

    template < typename MssDescriptorArray >
    struct reduction_data< MssDescriptorArray, false > {
        typedef notype reduction_type_t;

        reduction_data(const reduction_type_t val) {}
        constexpr reduction_type_t reduced_value() const { return 0; }
        reduction_type_t initial_value() const { return 0; }

        void assign(uint_t, const reduction_type_t &) {}
        void reduce() {}
    };

    template < typename MssDescriptorArray >
    struct reduction_data< MssDescriptorArray, true > {
        GRIDTOOLS_STATIC_ASSERT(
            (is_meta_array_of< MssDescriptorArray, is_amss_descriptor >::value), "Internal Error: wrong type");

        typedef typename boost::mpl::fold< typename MssDescriptorArray::elements,
            boost::mpl::vector0<>,
            boost::mpl::eval_if< amss_descriptor_is_reduction< boost::mpl::_2 >,
                                               boost::mpl::push_back< boost::mpl::_1, boost::mpl::_2 >,
                                               boost::mpl::_1 > >::type reduction_descriptor_seq_t;

        GRIDTOOLS_STATIC_ASSERT(
            (boost::mpl::size< reduction_descriptor_seq_t >::value == 1), "Error: more than one reduction found");

        typedef typename boost::mpl::front< reduction_descriptor_seq_t >::type reduction_descriptor_t;

        typedef typename reduction_descriptor_type< reduction_descriptor_t >::type reduction_type_t;

        typedef typename reduction_descriptor_t::bin_op_t bin_op_t;

        reduction_data(const reduction_type_t val)
            : m_initial_value(val), m_parallel_reduced_val(omp_get_max_threads(), val) {}
        const reduction_type_t &initial_value() const { return m_initial_value; }
        const reduction_type_t &parallel_reduced_val(const int elem) const { return m_parallel_reduced_val[elem]; }

        void assign(uint_t elem, const reduction_type_t &reduction_value) {
            assert(elem < m_parallel_reduced_val.size());
            m_parallel_reduced_val[elem] = bin_op_t()(m_parallel_reduced_val[elem], reduction_value);
        }

        void reduce() {
            m_reduced_value = m_initial_value;
#ifdef CXX11_ENABLED
            for (auto val : m_parallel_reduced_val) {
                m_reduced_value = bin_op_t()(m_reduced_value, val);
            }
#else
            for (uint_t i = 0; i < m_parallel_reduced_val.size(); ++i) {
                m_reduced_value = bin_op_t()(m_reduced_value, m_parallel_reduced_val[i]);
            }
#endif
        }
        reduction_type_t reduced_value() const { return m_reduced_value; }

      private:
        std::vector< reduction_type_t > m_parallel_reduced_val;
        reduction_type_t m_initial_value;
        reduction_type_t m_reduced_value;
    };

    template < typename T >
    struct is_reduction_data;

    template < typename MssDescriptorArray, bool HasReduction >
    struct is_reduction_data< reduction_data< MssDescriptorArray, HasReduction > > : boost::mpl::true_ {};
}
