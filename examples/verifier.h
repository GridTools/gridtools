#pragma once

class verifier
{
public:
    verifier(double precision) : precision_(precision) {}
    ~verifier(){}

    template<typename storage_type>
    bool verify(storage_type& field1, storage_type& field2)
    {
        assert(field1.template dims<0>() == field2.template dims<0>());
        assert(field1.template dims<1>() == field2.template dims<1>());
        assert(field1.template dims<2>() == field2.template dims<2>());

        const int idim = field1.template dims<0>();
        const int jdim = field1.template dims<1>();
        const int kdim = field1.template dims<2>();

        for(int i=0; i < idim; ++i)
        {
            for(int j=0; j < jdim; ++j)
            {
                for(int k=0; k < kdim; ++k)
                {
                    typename storage_type::value_type expected = field1(i,j,k);
                    typename storage_type::value_type actual = field2(i,j,k);

                    if(!compare_below_threashold(expected, actual))
                    {
                        std::cout << "Error in position " << i << " " << j << " " << k << " ; expected : " << expected <<
                                " ; actual : " << actual << std::endl;
                        return false;
                    }
                }
            }
        }
        return true;
    }
private:
    template<typename value_type>
    bool compare_below_threashold(value_type expected, value_type actual)
    {
        if (std::fabs(expected) < 1.0 && std::fabs(actual) < 1.0)
        {
            if(std::fabs(expected-actual) < precision_) return true;
        }
        else
        {
            if(std::fabs((expected-actual)/(precision_*expected)) < 1.0) return true;
        }
        return false;
    }
    double precision_;
};
