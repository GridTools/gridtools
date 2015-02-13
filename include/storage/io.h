#pragma once
#include <H5Cpp.h>
#include <gridtools.h>
#include <common/defs.h>

namespace gridtools{

#ifdef CXX11_ENABLED

    template<typename Storage>
    class hdf5_driver{
    public:

        hdf5_driver(char* file_handler, char* dataset_name, Storage const& storage):m_name(/*file_handler*/"out"), m_dataset_name(/*dataset_name*/"test"), m_file(/*m_name*/"out", H5F_ACC_TRUNC), m_storage(storage){
        }

        int_t  write(typename Storage::pointer_type data){
            // Try block to detect exceptions raised by any of the calls inside it
            // try
            // {
                /*
                 * Turn off the auto-printing when failure occurs so that we can
                 * handle the errors appropriately
                 */
                H5::Exception::dontPrint();

                /*
                 * Define the size of the array and create the data space for fixed
                 * size dataset.
                 */
                // hsize_t dimsf[Storage::space_dimensions]; // dataset dimensions
                //hsize_t dimsf[]={m_storage.size()}; // dataset dimensions
                hsize_t dimsf[]={100}; // dataset dimensions
                int vec[100]={0};
                /**
                   assigne the i,j,k storage dimensions
                 */
                // for(uint_t i=0; i<Storage::space_dimensions; ++i)
                //     dimsf[i]=m_storage.dims(i);

                // H5::DataSpace dataspace( Storage::space_dimensions, dimsf );
                H5::DataSpace dataspace( 1, dimsf );
                /*
                 * Define datatype for the data in the file.
                 * We will store little endian INT numbers.
                 */
                H5::IntType datatype( H5::PredType::NATIVE_INT );
                // H5::FloatType datatype( H5::PredType::NATIVE_DOUBLE );
                // datatype.setOrder( H5T_ORDER_LE );

                /*
                 * Create a new dataset within the file using defined dataspace and
                 * datatype and default dataset creation properties.
                 */
                H5::DataSet dataset = m_file.createDataSet( m_dataset_name, datatype, dataspace );
                /*
                 * Write the data to the dataset using default memory space, file
                 * space, and transfer properties.
                 */
                // dataset.write( data.get(), H5::PredType::NATIVE_DOUBLE );
                // dataset.write( data.get(), H5::PredType::NATIVE_INT );
                dataset.write( vec, H5::PredType::NATIVE_INT );
            // } catch( H5::FileIException error )
            // {
            //     error.printError();
            //     return -1;
            // }
            // // catch failure caused by the DataSet operations
            // catch( H5::DataSetIException error )
            // {
            //     error.printError();
            //     return -1;
            // }
            // // catch failure caused by the DataSpace operations
            // catch( H5::DataSpaceIException error )
            // {
            //     error.printError();
            //     return -1;
            // }
            // // catch failure caused by the DataSpace operations
            // catch( H5::DataTypeIException error )
            // {
            //     error.printError();
            //     return -1;
            // }
        }

    private:
        char* m_name;
        char* m_dataset_name;
        H5::H5File m_file;
        Storage const& m_storage;
    };
#endif
}//namespace gridtools
