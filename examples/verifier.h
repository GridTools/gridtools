#pragma once

namespace gridtools{

template < typename Storage, typename Partitioner >
class parallel_storage;
}

class verifier
{
public:
    verifier(const double precision, const int halo_size) : m_precision(precision), m_halo_size(halo_size) {}
    ~verifier(){}

    template<typename storage_type>
    bool verify(storage_type& field1, storage_type& field2)
    {
        assert(field1.template dims<0>() == field2.template dims<0>());
        assert(field1.template dims<1>() == field2.template dims<1>());
        assert(field1.template dims<2>() == field2.template dims<2>());

        const gridtools::uint_t idim = field1.template dims<0>();
        const gridtools::uint_t jdim = field1.template dims<1>();
        const gridtools::uint_t kdim = field1.template dims<2>();

        bool verified = true;

        for(gridtools::uint_t f=0; f<storage_type::field_dimensions; ++f)
            for(gridtools::uint_t i=m_halo_size; i < idim-m_halo_size; ++i)
            {
                for(gridtools::uint_t j=m_halo_size; j < jdim-m_halo_size; ++j)
                {
                    for(gridtools::uint_t k=0; k < kdim; ++k)
                    {
                        typename storage_type::value_type expected = field1.fields()[f][field1._index(i,j,k)];
                        typename storage_type::value_type actual = field2.fields()[f][field2._index(i,j,k)];

                        if(!compare_below_threashold(expected, actual))
                        {
                            std::cout << "Error in position " << i << " " << j << " " << k << " ; expected : " << expected <<
                                " ; actual : " << actual << "  " << std::fabs((expected-actual)/(expected))  << std::endl;
                            verified = false;
                        }
                    }
                }
            }

        return verified;
    }

    template<typename Partitioner, typename storage_type>
    bool verify(gridtools::parallel_storage<storage_type, Partitioner>& field1, storage_type& field2)
    {
        const gridtools::uint_t idim = field2.template dims<0>();
        const gridtools::uint_t jdim = field2.template dims<1>();
        const gridtools::uint_t kdim = field2.template dims<2>();

        bool verified = true;

        for(gridtools::uint_t f=0; f<storage_type::field_dimensions; ++f)
            for(gridtools::uint_t i=m_halo_size; i < idim-m_halo_size; ++i)
            {
                for(gridtools::uint_t j=m_halo_size; j < jdim-m_halo_size; ++j)
                {
                    for(gridtools::uint_t k=0; k < kdim; ++k)
                    {
                        if(field1.mine(i,j,k)){
                            typename storage_type::value_type expected = field2.get_value(i,j,k);
                            typename storage_type::value_type actual = field1.get_value(i,j,k);

                            if(!compare_below_threashold(expected, actual))
                            {
                                std::cout << "Error in position " << i << " " << j << " " << k << " ; expected : " << expected <<
                                    " ; actual : " << actual << "  " << std::fabs((expected-actual)/(expected))  << std::endl;
                                verified = false;
                            }
                        }
                    }
                }
            }

        return verified;
    }


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
    int m_halo_size;
};
