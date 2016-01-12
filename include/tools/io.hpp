#include <cstring>
#include <memory>
#include <Xdmf.hpp>
#include <XdmfReader.hpp>
#include <XdmfWriter.hpp>
#include <XdmfDomain.hpp>
#include <XdmfRectilinearGrid.hpp>
#include <XdmfRegularGrid.hpp>
#include <XdmfHDF5Writer.hpp>
#include <XdmfArray.hpp>
#include <XdmfAttribute.hpp>
#include <XdmfAttributeCenter.hpp>
#include <XdmfAttributeType.hpp>
#include <XdmfArrayType.hpp>
#include <XdmfHDF5Controller.hpp>
#include <XdmfInformation.hpp>
#include <XdmfTime.hpp>
#include <gridtools.hpp>

namespace gridtools{

    template<typename Derived>
    struct io_base;

    template<template < typename ... > class Derived, typename Storage,  typename ... Args >
    struct io_base<Derived<Storage, Args ... > > {

        io_base(Storage const& storage_):
            m_root(XdmfDomain::New())
            , m_storage(storage_)
        {
        }

        int read(std::string const& filename){
                    boost::shared_ptr<XdmfReader> reader = XdmfReader::New();
        boost::shared_ptr<XdmfDomain> domain =
            shared_dynamic_cast<XdmfDomain>(reader->read(filename));

        // boost::shared_ptr<XdmfRegularGrid> grid = domain->getRegularGrid(0);
        }

        int write(std::string const& filename){

        shared_ptr<XdmfHDF5Writer> exampleHeavyWriter = XdmfHDF5Writer::New(filename+".h5", true /*replace*/ );
        shared_ptr<XdmfWriter> exampleWriter = XdmfWriter::New(filename+".xmf", exampleHeavyWriter);

        //exampleHeavyWriter->setFileSizeLimit(1);


        // insert adds the information we just created as a leaf
        // Dimensions

        m_root->accept(exampleHeavyWriter);
        exampleHeavyWriter->setMode(XdmfHeavyDataWriter::Overwrite);//do this so that the data isn't in the hdf5 file twice.
        m_root->accept(exampleWriter);

        }

    protected:

        boost::shared_ptr<XdmfDomain>  m_root;
        Storage const& m_storage;
    };



    template<enumtype::grid_type>
    struct create_grid;

    template <>
    struct create_grid<enumtype::regular>{

        template<typename Storage>
        static boost::shared_ptr<XdmfRegularGrid> instance(Storage const& storage_){
            uint_t d1=storage_.meta_data().template dims<0>();
            uint_t d2=storage_.meta_data().template dims<1>();
            uint_t d3=storage_.meta_data().template dims<2>();
            uint_t first_dim = Storage::layout::template find_val<0, uint_t, 0>(d1, d2, d3);
            uint_t second_dim = Storage::layout::template find_val<1, uint_t, 0>(d1, d2, d3);
            uint_t third_dim = Storage::layout::template find_val<2, uint_t, 0>(d1, d2, d3);
            boost::shared_ptr<XdmfArray> dimensions = XdmfArray::New();
            dimensions->pushBack(first_dim);
            dimensions->pushBack(second_dim);
            dimensions->pushBack(third_dim);
            // Origin
            boost::shared_ptr<XdmfArray> origin = XdmfArray::New();
            origin->pushBack((unsigned int)0);
            origin->pushBack((unsigned int)0);
            origin->pushBack((unsigned int)0);

            // Brick Size
            uint_t first_size = Storage::layout::template find_val<0, uint_t, 0>(1, 10, 10);
            uint_t second_size = Storage::layout::template find_val<1, uint_t, 0>(1, 10, 10);
            uint_t third_size = Storage::layout::template find_val<2, uint_t, 0>(1, 10, 10);
            boost::shared_ptr<XdmfArray> brick = XdmfArray::New();
            brick->pushBack(third_size);
            brick->pushBack(second_size);
            brick->pushBack(first_size);
            return XdmfRegularGrid::New(brick, dimensions, origin);
        }
    };



    template <>
    struct create_grid<enumtype::rectilinear>{

        template <typename Storage, typename LocalGrid>
        static boost::shared_ptr<XdmfRectilinearGrid> instance(Storage const& storage_, LocalGrid const& local_grid_info_){

            uint_t d1=storage_.meta_data().template dims<0>();
            uint_t d2=storage_.meta_data().template dims<1>();
            uint_t d3=storage_.meta_data().template dims<2>();

            uint_t np1=local_grid_info_.template dims<0>();//n. local points along x
            uint_t np2=local_grid_info_.template dims<1>();//n. local points along y
            uint_t np3=local_grid_info_.template dims<2>();//n. local points along z

            uint_t first_dim = Storage::layout::template find_val<0, uint_t, 0>(d1, d2, d3);

            boost::shared_ptr<XdmfArray> coordinates1 = XdmfArray::New();
            boost::shared_ptr<XdmfArray> coordinates2 = XdmfArray::New();
            boost::shared_ptr<XdmfArray> coordinates3 = XdmfArray::New();

            double points1[d1*np1];// = {0,1,2,3,4,5};
            for(int_t i=0 ; i<d1 ; ++i)
            {
                for(int_t l=0 ; l<np1 ; ++l){
                    points1[i*np1+l] = storage_(i,0,0,l);
                    // std::cout<<" points1: "<<points1[i*np1+l];
                }
            }
             // std::cout<<std::endl;
            double points2[d2*np1];// = {0,1,2,3,4,5};
            for(int_t i=0 ; i<d2 ; ++i)
            {
                for(int_t l=0 ; l<np2 ; ++l){
                    points2[i*np2+l] = storage_(0,i,0,l*np1);
                    // std::cout<<" points2: "<<points2[i*np2+l];
                }
            }
            // std::cout<<std::endl;
            double points3[d3*np1];// = {0,1,2,3,4,5,6,7,8};
            for(int_t i=0 ; i<d3 ; ++i)
            {
                for(int_t l=0 ; l<np3 ; ++l){
                    points3[i*np3+l] = storage_(0,0,i,l*np1*np2);
                    // std::cout<<" points3: "<<points3[i*np3+l];
                }
            }
            // std::cout<<std::endl;

            coordinates1->insert(0, points1, d1*np1, 1, 1);
            coordinates2->insert(0, points2, d2*np2, 1, 1);
            coordinates3->insert(0, points3, d3*np3, 1, 1);

            return XdmfRectilinearGrid::New(coordinates1, coordinates2, coordinates3);
        }
    };


    template <typename Storage, typename LocalGridInfo>
    void reindex(Storage const& storage_, LocalGridInfo const& local_grid_info_, typename Storage::value_type * data_){
        auto d1=storage_.meta_data().template dims<0>();
        auto d2=storage_.meta_data().template dims<1>();
        auto d3=storage_.meta_data().template dims<2>();
        auto d4=storage_.meta_data().template dims<3>();
        auto d5=1;
        if(Storage::space_dimensions==5)
            d5 = storage_.meta_data().template dims<4>();//space dimension


        uint_t np1=local_grid_info_.template dims<0>();//n. local points along x
        uint_t np2=local_grid_info_.template dims<1>();//n. local points along y
        uint_t np3=local_grid_info_.template dims<2>();//n. local points along z

        for(int_t k=0 ; k<d3 ; ++k)
        {
            for(int_t n=0 ; n<np3 ; ++n)
            {
                for(int_t j=0 ; j<d2 ; ++j)
                {
                    for(int_t m=0 ; m<np2 ; ++m)
                    {
                        for(int_t i=0 ; i<d1 ; ++i)
                        {
                            for(int_t l=0 ; l<np1 ; ++l)
                            {
                                for(int_t dim=0 ; dim<d5 ; ++dim)
                                {
                                    data_[ k*np3*d2*np2*d1*np1*d5 + n*d2*np2*d1*np1*d5 + j*np2*d1*np1*d5 + m*d1*np1*d5 + i*np1*d5 + l*d5 + dim] = storage_(i,j,k,l+np1*m+np1*np2*n,dim);
                                //std::cout<<"("<<i<<","<<j<<","<<k<<","<<l<<","<<m<<","<<n<<") = ("<<storage_(i,j,k,l,0)<<","<<storage_(i,j,k,l,1)<<","<<storage_(i,j,k,l,2)<<")"<<std::endl;
                                }
                            }
                        }
                    }

                }
            }
        }
    }

    template<typename ... Storage>
    struct io_regular ;

    template<typename Storage>
    struct io_regular<Storage> : public io_base<io_regular<Storage> > {

        typedef Storage storage_t;
        typedef io_base<io_regular<Storage> > super;

        io_regular(Storage const& storage_):
            super(storage_)
            , m_grid(create_grid<enumtype::regular>::instance(storage_))
        {
        }

        template<int FieldDim=0 >//cxx11
        void set_attribute(Storage const& storage_,  std::string const& name_){

            // Attribute
            boost::shared_ptr<XdmfAttribute> attr = XdmfAttribute::New();
            attr->setName(name_);
            attr->setCenter(XdmfAttributeCenter::Node());
            attr->setType(XdmfAttributeType::Scalar());
            uint_t total_points = storage_.meta_data().template dims<2>()*storage_.meta_data().template dims<1>()*storage_.meta_data().template dims<0>();
            attr->initialize(XdmfArrayType::Float64(), total_points);
            attr->insert(0, storage_.template access_value<static_int<FieldDim> >(), total_points);
            // The heavy data set name is determined by the writer if not set
            m_grid->insert(attr);
        }

        boost::shared_ptr< XdmfRegularGrid> m_grid;

    };


    template<typename ... Storage>
    struct io_rectilinear ;

    template<typename Storage, typename LocalGrid>
    struct io_rectilinear<Storage, LocalGrid> : public io_base<io_rectilinear<Storage, LocalGrid> > {

        typedef Storage storage_t;

        typedef io_base<io_rectilinear<Storage, LocalGrid> > super;
        io_rectilinear(Storage const& storage_, LocalGrid const& local_grid_):
            super(storage_)
            , m_local_grid(local_grid_)
            , m_grid(create_grid<enumtype::rectilinear>::instance(storage_, local_grid_))
        {
            // Grid
            m_grid->setName("Regular Structured Grid");
            this->m_root->insert(m_grid);

        }


        template<int FieldDim >//cxx11
        void set_attribute(Storage const& storage_, std::string const& name_){

            // Attribute
            boost::shared_ptr<XdmfAttribute> attr = XdmfAttribute::New();
            attr->setName(name_);
            attr->setCenter(XdmfAttributeCenter::Node());
            attr->setType(XdmfAttributeType::Scalar());
            uint_t total_points = storage_.meta_data().size();
            typename Storage::value_type data[total_points];
            reindex(storage_, m_local_grid, data); // loops
            attr->initialize(XdmfArrayType::Float64(), total_points);
            attr->insert(0, data, total_points);
            // The heavy data set name is determined by the writer if not set
            m_grid->insert(attr);
        }

        template<int FieldDim, typename VecStorage >//cxx11
        void set_attribute(VecStorage const& storage_, std::string const& name_){

            // Attribute
            boost::shared_ptr<XdmfAttribute> attr = XdmfAttribute::New();
            attr->setName(name_);
            attr->setCenter(XdmfAttributeCenter::Node());
            attr->setType(XdmfAttributeType::Vector());
            uint_t total_points = storage_.meta_data().size();
            typename Storage::value_type data[total_points];
            reindex(storage_, m_local_grid, data); // loops

            attr->initialize(XdmfArrayType::Float64(), total_points);
            attr->insert(0, data, total_points);
            // The heavy data set name is determined by the writer if not set
            m_grid->insert(attr);
        }


        void set_information(std::string const& name_){
            // Information
            boost::shared_ptr<XdmfInformation>  i = XdmfInformation::New(); //# Arbitrary Name=Value Facility
            // i->setName("Time");
            i->setValue(name_);
            this->m_root->insert(i);  // XdmfDomain is the root of the tree

        }

        boost::shared_ptr< XdmfRectilinearGrid> m_grid;
    private:
        LocalGrid const& m_local_grid;
    };



}//namespace gridtools
