#pragma once

class verifier
{
public:
    verifier(const double precision, const int halo_size, const int i_size, const int j_size, const int k_size) :
        m_precision(precision), m_halo_size(halo_size), m_i_size(i_size), m_j_size(j_size), m_k_size(k_size) {}
    ~verifier(){}

    template<typename storage_type>
    bool verify(storage_type& field1, storage_type& field2)
    {
        bool verified = true;
        for(int i=m_halo_size; i < m_i_size-m_halo_size; ++i)
        {
            for(int j=m_halo_size; j < m_j_size-m_halo_size; ++j)
            {
                for(int k=0; k < m_k_size; ++k)
                {
                    typename storage_type::value_type expected = field1(i,j,k);
                    typename storage_type::value_type actual = field2(i,j,k);

                    if(!compare_below_threashold(expected, actual))
                    {
                        std::cout << "Error in position " << i << " " << j << " " << k << " ; expected : " << expected <<
                                " ; actual : " << actual << std::endl;
                        verified = false;
                    }
                }
            }
        }

        return verified;
    }
#ifdef CXX11_ENABLED

    template <short_t field_dim1, short_t snapshot1, short_t field_dim2, short_t snapshot2, typename First,  typename  ...  StorageExtended>
    bool verify(gridtools::extend_dim<First, StorageExtended...>& field1, gridtools::extend_dim<First, StorageExtended...>& field2)
    {
        typedef gridtools::extend_dim<First, StorageExtended...> storage_type;
        typedef typename storage_type::original_storage::value_type value_type;

        bool verified = true;
        for(int i=m_halo_size; i < m_i_size-m_halo_size; ++i)
        {
            for(int j=m_halo_size; j < m_j_size-m_halo_size; ++j)
            {
                for(int k=0; k < m_k_size; ++k)
                {
                    value_type expected = *(field1.template get<field_dim1,snapshot1>(i,j,k));
                    value_type actual = *(field2.template get<field_dim2,snapshot2>(i,j,k));

                    if(!compare_below_threashold(expected, actual))
                    {
                        std::cout << "Error in position " << i << " " << j << " " << k << " ; expected : " << expected <<
                                " ; actual : " << actual << std::endl;
                        verified = false;
                    }
                }
            }
        }

        return verified;
    }
#endif

private:
    template<typename value_type>
    bool compare_below_threashold(value_type expected, value_type actual)
    {
        if (std::fabs(expected) < 1e-3 && std::fabs(actual) < 1e-3)
        {
            if(std::fabs(expected-actual) < m_precision) return true;
        }
        else
        {
            if(std::fabs((expected-actual)/(m_precision*expected)) < 1.0) return true;
        }
        return false;
    }
    double m_precision;
    const int m_halo_size, m_i_size, m_j_size, m_k_size;
};
