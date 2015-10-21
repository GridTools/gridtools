#pragma once

namespace gridtools{

    template < typename Partitioner, typename Storage >
    class parallel_storage_info;
}

class verifier
{
public:
    verifier(const double precision, const int halo_size) : m_precision(precision), m_halo_size(halo_size) {}
    ~verifier(){}

    template<typename storage_type>
    bool verify(storage_type const& field1, storage_type const& field2) const
    {
        // assert(field1.template dims<0>() == field2.template dims<0>());
        // assert(field1.template dims<1>() == field2.template dims<1>());
        // assert(field1.template dims<2>() == field2.template dims<2>());
        typename storage_type::meta_data_t const* meta=&field1.meta_data();

        const gridtools::uint_t idim = meta->template dims<0>();
        const gridtools::uint_t jdim = meta->template dims<1>();
        const gridtools::uint_t kdim = meta->template  dims<2>();

        bool verified = true;

        for(gridtools::uint_t f=0; f<storage_type::field_dimensions; ++f)
            for(gridtools::uint_t i=m_halo_size; i < idim-m_halo_size; ++i)
            {
                for(gridtools::uint_t j=m_halo_size; j < jdim-m_halo_size; ++j)
                {
                    for(gridtools::uint_t k=0; k < kdim; ++k)
                    {
                        typename storage_type::value_type expected = field1.fields()[f][meta->index(i,j,k)];
                        typename storage_type::value_type actual = field2.fields()[f][meta->index(i,j,k)];

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

    template<typename Partitioner, typename MetaStorageType, typename StorageType>
    bool verify_parallel(gridtools::parallel_storage_info<MetaStorageType, Partitioner> const& metadata_, StorageType const& field1, StorageType const& field2)
    {

        const gridtools::uint_t idim = metadata_.get_metadata().template dims<0>();
        const gridtools::uint_t jdim = metadata_.get_metadata().template dims<1>();
        const gridtools::uint_t kdim = metadata_.get_metadata().template dims<2>();

        bool verified = true;

        for(gridtools::uint_t f=0; f<StorageType::field_dimensions; ++f)
            for(gridtools::uint_t i=m_halo_size; i < idim-m_halo_size; ++i)
            {
                for(gridtools::uint_t j=m_halo_size; j < jdim-m_halo_size; ++j)
                {
                    for(gridtools::uint_t k=0; k < kdim; ++k)
                    {
                        if(metadata_.mine(i,j,k)){
                            typename StorageType::value_type expected = field2.get_value(i,j,k);
                            typename StorageType::value_type actual = field1[metadata_.get_local_index(i,j,k)];

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
    bool compare_below_threashold(value_type expected, value_type actual) const
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
