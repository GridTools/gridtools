#pragma once
#include "common/array.hpp"
#include "common/array_addons.hpp"

namespace gridtools{

    template < typename Partitioner, typename Storage >
    class parallel_storage_info;

    template<typename value_type>
    bool compare_below_threshold(value_type expected, value_type actual, double precision)
    {
        if (std::fabs(expected) < 1e-3 && std::fabs(actual) < 1e-3)
        {
            if(std::fabs(expected-actual) < precision) return true;
        }
        else
        {
            if(std::fabs((expected-actual)/(precision*expected)) < 1.0) return true;
        }
        return false;
    }

#ifdef CXX11_ENABLED

    template<uint_t NDim, uint_t NCoord, typename StorageType>
    struct verify_helper
    {
        verify_helper(StorageType const & exp_field, StorageType const & actual_field, uint_t field_id,
                      array<array<uint_t, 2>, StorageType::space_dimensions> const & halos, double precision) :
            m_exp_field(exp_field),
            m_actual_field(actual_field),
            m_field_id(field_id),
            m_precision(precision),
            m_halos(halos){}

        template <typename Coords>
        bool operator()(Coords const& coords_, array<uint_t, NCoord> const &pos)
        {
            typename StorageType::meta_data_t const* meta=&m_exp_field.meta_data();

            const gridtools::uint_t size = meta->template  dims<NDim-1>();
            bool verified = true;

            verify_helper<NDim-1, NCoord+1, StorageType> next_loop(m_exp_field, m_actual_field, m_field_id, m_halos, m_precision);

            const uint_t halo_minus = m_halos[NDim-1][0];
            const uint_t halo_plus = m_halos[NDim-1][1];

            for(int c=halo_minus; c < size-halo_plus; ++c)
            {
                array<uint_t,NCoord+1> new_pos = pos.prepend_dim(c);
                verified = verified & next_loop(coords_, new_pos);
            }
            return verified;
        }

    private:
        StorageType const& m_exp_field;
        StorageType const& m_actual_field;
        double m_precision;
        uint_t m_field_id;
        array<array<uint_t, 2>, StorageType::space_dimensions> const& m_halos;
    };

    template<uint_t NCoord, typename StorageType>
    struct verify_helper<0,NCoord, StorageType>
    {
        verify_helper(StorageType const & exp_field, StorageType const & actual_field, uint_t field_id,
                      array<array<uint_t, 2>, StorageType::space_dimensions> const & halos, double precision) :
            m_exp_field(exp_field),
            m_actual_field(actual_field),
            m_field_id(field_id),
            m_precision(precision),
            m_halos(halos){}

        template<typename Coords>
        bool operator()(Coords const& coords_, array<uint_t, NCoord> const &pos)
        {
            bool verified = true;
            /* hypothesis: coordinate 2 is along k */
            if(pos[2] < coords_.value_at_top())
            {
                typename StorageType::meta_data_t const* meta=&m_exp_field.meta_data();

                typename StorageType::value_type expected = m_exp_field.fields()[m_field_id][meta->index(pos)];
                typename StorageType::value_type actual = m_actual_field.fields()[m_field_id][meta->index(pos)];
                if(!compare_below_threshold(expected, actual, m_precision))
                {
                    std::cout << "Error in field dimension " << m_field_id << " and position " << pos << " ; expected : " << expected <<
                        " ; actual : " << actual << "  " << std::fabs((expected-actual)/(expected))  << std::endl;
                    verified = false;
                }
            }
            return verified;
        }
    private:
        StorageType const& m_exp_field;
        StorageType const& m_actual_field;
        uint_t m_field_id;
        double m_precision;
        array<array<uint_t, 2>, StorageType::space_dimensions> const& m_halos;
    };


    template<uint_t NDim, typename Coords, typename StorageType>
    bool verify_functor(Coords const& coords_, StorageType const & exp_field, StorageType const & actual_field, uint_t field_id,
                        array<array<uint_t, 2>, StorageType::space_dimensions> halos, double precision)
    {
        typename StorageType::meta_data_t const* meta=&exp_field.meta_data();

        const gridtools::uint_t size = meta->template  dims<NDim-1>();
        bool verified = true;
        verify_helper<NDim-1, 1, StorageType> next_loop(exp_field, actual_field, field_id, halos, precision);

        const uint_t halo_minus = halos[NDim-1][0];
        const uint_t halo_plus = halos[NDim-1][1];

        for(uint_t c=halo_minus; c < size-halo_plus; ++c)
        {
#ifdef CXX11_ENABLED
            array<uint_t,1> new_pos{c};
#else
            array<uint_t,1> new_pos(c);
#endif
            verified = verified & next_loop(coords_, new_pos);
        }
        return verified;

    }

    class verifier
    {
    public:
        verifier(const double precision) : m_precision(precision) {}
        ~verifier(){}

        template<typename Coords, typename StorageType>
        bool verify(Coords const& coords_, StorageType const& field1, StorageType const& field2, const array<array<uint_t, 2>, StorageType::space_dimensions> halos)
        {
            typename StorageType::meta_data_t const* meta=&field1.meta_data();

            bool verified = true;

            for(gridtools::uint_t f=0; f<StorageType::field_dimensions; ++f)
            {
                verified=verify_functor<StorageType::space_dimensions>(coords_, field1, field2, f, halos, m_precision);
            }
            return verified;
        }

        template<typename Coords, typename Partitioner, typename MetaStorageType, typename StorageType>
        bool verify_parallel(Coords const& coords_, gridtools::parallel_storage_info<MetaStorageType, Partitioner> const& metadata_, StorageType const& field1, StorageType const& field2,
                              const array<array<uint_t, 2>, StorageType::space_dimensions> halos)
        {

            const gridtools::uint_t idim = metadata_.get_metadata().template dims<0>();
            const gridtools::uint_t jdim = metadata_.get_metadata().template dims<1>();
            const gridtools::uint_t kdim = metadata_.get_metadata().template dims<2>();

            bool verified = true;

            for(gridtools::uint_t f=0; f<StorageType::field_dimensions; ++f)
                for(gridtools::uint_t i=halos[0][0]; i < idim-halos[0][1]; ++i)
                {
                    for(gridtools::uint_t j=halos[1][0]; j < jdim-halos[1][1]; ++j)
                    {
                        for(gridtools::uint_t k=0; k < coords_.value_at_top(); ++k)
                        {
                            if(metadata_.mine(i,j,k)){
                                typename StorageType::value_type expected = field2.get_value(i,j,k);
                                typename StorageType::value_type actual = field1[metadata_.get_local_index(i,j,k)];

                                if(!compare_below_threshold(expected, actual, m_precision))
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
        double m_precision;
    };
#else
    class verifier
    {
    public:
        verifier(const double precision, const int halo_size) : m_precision(precision), m_halo_size(halo_size) {}
        ~verifier(){}

        template<typename Coords, typename storage_type>
        bool verify(Coords const& coords_, storage_type const& field1, storage_type const& field2) const
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
                        for(gridtools::uint_t k=0; k < coords_.value_at_top(); ++k)
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

        template<typename Coords, typename Partitioner, typename MetaStorageType, typename StorageType>
        bool verify_parallel(Coords const& coords_, gridtools::parallel_storage_info<MetaStorageType, Partitioner> const& metadata_, StorageType const& field1, StorageType const& field2)
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
                        for(gridtools::uint_t k=0; k < coords_.value_at_top(); ++k)
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
#endif

} // namespace gridtools
