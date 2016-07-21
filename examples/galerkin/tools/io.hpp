#pragma once
#include <cstring>
#ifndef __CUDACC__
#include <memory>
#endif
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
#include <iostream>
#include <fstream>

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

        return 0;
        }

    protected:

        boost::shared_ptr<XdmfDomain>  m_root;
        Storage const& m_storage;
    };



    template<enumtype::grid_type>
    struct create_grid;

    /**@brief creates a regular grid

       This grid must be fully structured hexahedral grid, with equally sized elements
    */
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


    /**@brief creates a rectilinear grid

       This grid must be fully structured hexahedral grid, but the elements can have different sizes
    */
    template <>
    struct create_grid<enumtype::rectilinear>{

        template <typename Storage, typename LocalGrid>
        static boost::shared_ptr<XdmfRectilinearGrid> instance(Storage const& storage_, LocalGrid const& local_grid_info_){

            uint_t d1=storage_.meta_data().template dims<0>();
            uint_t d2=storage_.meta_data().template dims<1>();
            uint_t d3=storage_.meta_data().template dims<2>();

            //quad points or dofs?
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
                    points1[i*np1+l] = storage_(i,0,0,l,0);
                    // std::cout<<" points1: "<<points1[i*np1+l];
                }
            }
             // std::cout<<std::endl;
            double points2[d2*np1];// = {0,1,2,3,4,5};
            for(int_t i=0 ; i<d2 ; ++i)
            {
                for(int_t l=0 ; l<np2 ; ++l){
                    points2[i*np2+l] = storage_(0,i,0,l*np1,1);
                    // std::cout<<" points2: "<<points2[i*np2+l];
                }
            }
            // std::cout<<std::endl;
            double points3[d3*np1];// = {0,1,2,3,4,5,6,7,8};
            for(int_t i=0 ; i<d3 ; ++i)
            {
                for(int_t l=0 ; l<np3 ; ++l){
                    points3[i*np3+l] = storage_(0,0,i,l*np1*np2,2);
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

    template<ushort_t CubDegree, enumtype::grid_type>
    struct create_grid_qpoints;

    /**@brief creates a rectilinear grid

       This grid must be fully structured hexahedral grid, but the elements can have different sizes
    */
    template <ushort_t CubDegree>
    struct create_grid_qpoints<CubDegree, enumtype::rectilinear>{

        template <typename Storage, typename Cubature>
        static boost::shared_ptr<XdmfRectilinearGrid> instance(Storage const& storage_, Cubature const& cub){

            uint_t d1=storage_.meta_data().template dims<0>();
            uint_t d2=storage_.meta_data().template dims<1>();
            uint_t d3=storage_.meta_data().template dims<2>();

            //quad points: only valid for tensor products of course
            uint_t np1=CubDegree - 1;//n. local points along x
            uint_t np2=CubDegree - 1;//n. local points along y
            uint_t np3=CubDegree - 1;//n. local points along z

            uint_t first_dim = Storage::layout::template find_val<0, uint_t, 0>(d1, d2, d3);

            boost::shared_ptr<XdmfArray> coordinates1 = XdmfArray::New();
            boost::shared_ptr<XdmfArray> coordinates2 = XdmfArray::New();
            boost::shared_ptr<XdmfArray> coordinates3 = XdmfArray::New();

            double points1[d1*np1];// = {0,1,2,3,4,5};
            for(int_t i=0 ; i<d1 ; ++i)
            {
                for(int_t l=0 ; l<np1 ; ++l){
                    points1[i*np1+l] = storage_(i,0,0,0,0)+cub(l,0,0);
                    // std::cout<<" points1: "<<points1[i*np1+l];
                }
            }

            // std::cout<<std::endl;
            double points2[d2*np1];// = {0,1,2,3,4,5};
            for(int_t i=0 ; i<d2 ; ++i)
            {
                for(int_t l=0 ; l<np2 ; ++l){
                    points2[i*np2+l] = storage_(0,i,0,0,1)+cub(l*np1,1,0);
                    // std::cout<<" points2: "<<points2[i*np2+l];
                }
            }

            // std::cout<<std::endl;
            double points3[d3*np1];// = {0,1,2,3,4,5,6,7,8};
            for(int_t i=0 ; i<d3 ; ++i)
            {
                for(int_t l=0 ; l<np3 ; ++l){
                    points3[i*np3+l] = storage_(0,0,i,0,2)+cub(l*np1*np2,2,0);
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


    /**
       @brief export a matrix in matrix market format

       This function writes a matrix to a text file in matrix market format.
       This format can be loaded by Matlab/Octave to see the matrix pattern,
       and it is a specific format for sparse matrices. It consists of lines with
       3 entries: the first 2 for the i,j coordinates, and the third for the nonzero matrix value.
     */
    template <typename Storage>
    void spy(Storage const& storage_, char const* name){

        std::ofstream o_file;
        o_file.open(name);
        auto d1=storage_.meta_data().template dims<0>();
        auto d2=storage_.meta_data().template dims<1>();
        auto d3=storage_.meta_data().template dims<2>();
        auto d4=storage_.meta_data().template dims<3>();
        auto d5 = storage_.meta_data().template dims<4>();

        o_file<<"%%MatrixMarket matrix coordinate real general\n";
        o_file<<d1*d2*d3*d4<<" "<<d1*d2*d3*d5<<" "<<d1*d2*d3*d4*d5<<"\n";
        for(int_t k=0 ; k<d3 ; ++k)
        {
            for(int_t j=0 ; j<d2 ; ++j)
            {
                for(int_t i=0 ; i<d1 ; ++i)
                {
                    uint_t offset_=i*d2*d3*d4+j*d3*d4+k*d4;
                    for(int_t l=0 ; l<d4 ; ++l)
                    {
                        for(int_t m=0 ; m<d5 ; ++m)
                        {
                            o_file<< offset_+l+1 <<" "<< offset_+m+1 <<" "<<storage_(i,j,k,l,m)<<"\n";
                        }
                    }
                }
            }
        }

        o_file.close();
    }

    /**
       @brief exports a vector

       this function writes out a vector in text format, in a newline-separated list
       which can be loaded (e.g.) by Matlab/Octave
     */
    template <typename Storage>
    void spy_vec(Storage const& storage_, char const* name){

        std::ofstream o_file;
        o_file.open(name);
        auto d1=storage_.meta_data().template dims<0>();
        auto d2=storage_.meta_data().template dims<1>();
        auto d3=storage_.meta_data().template dims<2>();
        auto d4=storage_.meta_data().template dims<3>();

        for(int_t i=0 ; i<d1 ; ++i)
        {
            for(int_t j=0 ; j<d2 ; ++j)
            {
                for(int_t k=0 ; k<d3 ; ++k)
                {
                    //uint_t offset_=i*d2*d3*d4+j*d3*d4+k*d4;
                    for(int_t l=0 ; l<d4 ; ++l)
                    {
                        o_file<<storage_(i,j,k,l)<<"\n";
                    }
                }
            }
        }

        o_file.close();
    }


    /**
       @brief function to serialize the scalar/vector valued fields living on a mesh into an array

       This is needed in order to write the vector to the output file
     */
    template <typename Storage, typename LocalGridInfo>
    void reindex_vec(Storage const& storage_, LocalGridInfo const& local_grid_info_, typename Storage::value_type * data_){
        auto d1=storage_.meta_data().template dims<0>();
        auto d2=storage_.meta_data().template dims<1>();
        auto d3=storage_.meta_data().template dims<2>();
        auto d4=storage_.meta_data().template dims<3>();
        auto d5=1;
        if(Storage::space_dimensions>=5)
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

    template <typename Storage, typename LocalGridInfo>
    void reindex(Storage const& storage_, LocalGridInfo const& local_grid_info_, typename Storage::value_type * data_){
        auto d1=storage_.meta_data().template dims<0>();
        auto d2=storage_.meta_data().template dims<1>();
        auto d3=storage_.meta_data().template dims<2>();
        auto d4=storage_.meta_data().template dims<3>();

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
                                    data_[ k*np3*d2*np2*d1*np1 + n*d2*np2*d1*np1 + j*np2*d1*np1 + m*d1*np1 + i*np1 + l] = storage_(i,j,k,l+np1*m+np1*np2*n);
                            }
                        }
                    }
                }
            }
        }
    }


    template <ushort_t CubDegree, typename Storage, typename Cubature>
    void reindex_on_qpoints(Storage const& storage_, Cubature const& cubature_, typename Storage::value_type * data_){
        auto d1=storage_.meta_data().template dims<0>();
        auto d2=storage_.meta_data().template dims<1>();
        auto d3=storage_.meta_data().template dims<2>();
        auto d4=storage_.meta_data().template dims<3>();

        // uint_t np1=local_grid_info_.template dims<0>();//n. local points along x
        // uint_t np2=local_grid_info_.template dims<1>();//n. local points along y
        // uint_t np3=local_grid_info_.template dims<2>();//n. local points along z
        uint_t np1=CubDegree - 1;//n. local points along x
        uint_t np2=CubDegree - 1;//n. local points along y
        uint_t np3=CubDegree - 1;//n. local points along z

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
                                    data_[ k*np3*d2*np2*d1*np1 + n*d2*np2*d1*np1 + j*np2*d1*np1 + m*d1*np1 + i*np1 + l] = storage_(i,j,k,l+np1*m+np1*np2*n);
                            }
                        }
                    }
                }
            }
        }
    }


    template <ushort_t Face, typename Storage, typename LocalGridInfo>
    void reindex_on_face(Storage const& storage_, LocalGridInfo const& local_grid_info_, typename Storage::value_type * data_){
        auto d1=storage_.meta_data().template dims<0>();
        auto d2=storage_.meta_data().template dims<1>();
        auto d3=storage_.meta_data().template dims<2>();
        auto d4=8;//storage_.meta_data().template dims<3>();
        auto d5=1;
        if(Storage::space_dimensions>=5)
            d5 = storage_.meta_data().template dims<3>();//space dimension

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
                                    if(Face==0)
                                        if(!l)
                                            data_[ k*np3*d2*np2*d1*np1*d5 + n*d2*np2*d1*np1*d5 + j*np2*d1*np1*d5 + m*d1*np1*d5 + i*np1*d5 + l*d5 + dim] = storage_(i,j,k,dim, 0);
                                    if(Face==2)
                                        if(!m)
                                            data_[ k*np3*d2*np2*d1*np1*d5 + n*d2*np2*d1*np1*d5 + j*np2*d1*np1*d5 + m*d1*np1*d5 + i*np1*d5 + l*d5 + dim] = storage_(i,j,k,dim, 2);
                                    if(Face==4)
                                        if(!n)
                                            data_[ k*np3*d2*np2*d1*np1*d5 + n*d2*np2*d1*np1*d5 + j*np2*d1*np1*d5 + m*d1*np1*d5 + i*np1*d5 + l*d5 + dim] = storage_(i,j,k,dim, 4);
                                    if(l && m && n)
                                        data_[ k*np3*d2*np2*d1*np1*d5 + n*d2*np2*d1*np1*d5 + j*np2*d1*np1*d5 + m*d1*np1*d5 + i*np1*d5 + l*d5 + dim] = 0.;
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


        template<int FieldDim, typename ScalarStorage >//cxx11
        void set_attribute_scalar(ScalarStorage const& storage_, std::string const& name_){

            // Attribute
            boost::shared_ptr<XdmfAttribute> attr = XdmfAttribute::New();
            attr->setName(name_);
            attr->setCenter(XdmfAttributeCenter::Node());
            attr->setType(XdmfAttributeType::Scalar());
            uint_t total_points = storage_.meta_data().size();
            typename ScalarStorage::value_type data[total_points];
            reindex(storage_, m_local_grid, data); // loops
            attr->initialize(XdmfArrayType::Float64(), total_points);
            attr->insert(0, data, total_points);
            // The heavy data set name is determined by the writer if not set
            m_grid->insert(attr);
        }

        template<int FieldDim, typename VecStorage >//cxx11
        void set_attribute_vector(VecStorage const& storage_, std::string const& name_){

            // Attribute
            boost::shared_ptr<XdmfAttribute> attr = XdmfAttribute::New();
            attr->setName(name_);
            attr->setCenter(XdmfAttributeCenter::Node());
            attr->setType(XdmfAttributeType::Vector());
            uint_t total_points = storage_.meta_data().size();
            typename VecStorage::value_type data[total_points];
            reindex_vec(storage_, m_local_grid, data); // loops

            attr->initialize(XdmfArrayType::Float64(), total_points);
            attr->insert(0, data, total_points);
            // The heavy data set name is determined by the writer if not set
            m_grid->insert(attr);
        }


        template<int FieldDim, typename VecStorage >//cxx11
        void set_attribute_vector_on_face(VecStorage const& storage_, std::string const& name_){

            // Attribute
            boost::shared_ptr<XdmfAttribute> attr0 = XdmfAttribute::New();
            attr0->setName(name_ + "face0");
            attr0->setCenter(XdmfAttributeCenter::Node());
            attr0->setType(XdmfAttributeType::Vector());
            uint_t total_points = storage_.meta_data().template dims<0>()
                *storage_.meta_data().template dims<1>()
                *storage_.meta_data().template dims<2>()
                *8//dofs cardinality
                *storage_.meta_data().template dims<3>();//n_dims
            typename VecStorage::value_type data[total_points];
            reindex_on_face<0>(storage_, m_local_grid, data); // loops

            attr0->initialize(XdmfArrayType::Float64(), total_points);
            attr0->insert(0, data, total_points);
            // The heavy data set name is determined by the writer if not set
            m_grid->insert(attr0);

            // Attr1ibute
            boost::shared_ptr<XdmfAttribute> attr1 = XdmfAttribute::New();
            attr1->setName(name_ + "face1");
            attr1->setCenter(XdmfAttributeCenter::Node());
            attr1->setType(XdmfAttributeType::Vector());
            typename VecStorage::value_type data1[total_points];
            reindex_on_face<1>(storage_, m_local_grid, data1); // loops

            attr1->initialize(XdmfArrayType::Float64(), total_points);
            attr1->insert(0, data, total_points);
            // The heavy data set name is determined by the writer if not set
            m_grid->insert(attr1);

            // Attribute
            boost::shared_ptr<XdmfAttribute> attr2 = XdmfAttribute::New();
            attr2->setName(name_ + "face2");
            attr2->setCenter(XdmfAttributeCenter::Node());
            attr2->setType(XdmfAttributeType::Vector());
            typename VecStorage::value_type data2[total_points];
            reindex_on_face<2>(storage_, m_local_grid, data2); // loops

            attr2->initialize(XdmfArrayType::Float64(), total_points);
            attr2->insert(0, data, total_points);
            // The heavy data set name is determined by the writer if not set
            m_grid->insert(attr2);

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




    template<typename ... Storage>
    struct io_rectilinear_qpoints ;

    template<typename Storage, typename LocalCub, typename CubDegree>
    struct io_rectilinear_qpoints<Storage, LocalCub, CubDegree> : public io_base<io_rectilinear_qpoints<Storage, LocalCub, CubDegree> > {

        typedef Storage storage_t;

        typedef io_base<io_rectilinear_qpoints<Storage, LocalCub, CubDegree> > super;
        io_rectilinear_qpoints(Storage const& storage_, LocalCub const& local_cub_):
            super(storage_)
            , m_local_cub(local_cub_)
            , m_grid(create_grid_qpoints<CubDegree::value, enumtype::rectilinear>::instance(storage_, local_cub_))
        {
            // Grid
            m_grid->setName("Regular Structured Grid");
            this->m_root->insert(m_grid);
        }


        template<int FieldDim, typename ScalarStorage>//cxx11
        void set_attribute_scalar(ScalarStorage const& storage_, std::string const& name_){

            // Attribute
            boost::shared_ptr<XdmfAttribute> attr = XdmfAttribute::New();
            attr->setName(name_);
            attr->setCenter(XdmfAttributeCenter::Node());
            attr->setType(XdmfAttributeType::Scalar());
            uint_t total_points = storage_.meta_data().size();
            typename ScalarStorage::value_type data[total_points];
            reindex_on_qpoints<CubDegree::value>(storage_, m_local_cub, data); // loops
            attr->initialize(XdmfArrayType::Float64(), total_points);
            attr->insert(0, data, total_points);
            // The heavy data set name is determined by the writer if not set
            m_grid->insert(attr);
        }

        // template<int FieldDim, typename VecStorage >//cxx11
        // void set_attribute_vector(VecStorage const& storage_, std::string const& name_){

        //     // Attribute
        //     boost::shared_ptr<XdmfAttribute> attr = XdmfAttribute::New();
        //     attr->setName(name_);
        //     attr->setCenter(XdmfAttributeCenter::Node());
        //     attr->setType(XdmfAttributeType::Vector());
        //     uint_t total_points = storage_.meta_data().size();
        //     typename VecStorage::value_type data[total_points];
        //     reindex_vec(storage_, m_local_cub, data); // loops

        //     attr->initialize(XdmfArrayType::Float64(), total_points);
        //     attr->insert(0, data, total_points);
        //     // The heavy data set name is determined by the writer if not set
        //     m_grid->insert(attr);
        // }


        // template<int FieldDim, typename VecStorage >//cxx11
        // void set_attribute_vector_on_face(VecStorage const& storage_, std::string const& name_){

        //     // Attribute
        //     boost::shared_ptr<XdmfAttribute> attr0 = XdmfAttribute::New();
        //     attr0->setName(name_ + "face0");
        //     attr0->setCenter(XdmfAttributeCenter::Node());
        //     attr0->setType(XdmfAttributeType::Vector());
        //     uint_t total_points = storage_.meta_data().template dims<0>()
        //         *storage_.meta_data().template dims<1>()
        //         *storage_.meta_data().template dims<2>()
        //         *8//dofs cardinality
        //         *storage_.meta_data().template dims<3>();//n_dims
        //     typename VecStorage::value_type data[total_points];
        //     reindex_on_face<0>(storage_, m_local_cub, data); // loops

        //     attr0->initialize(XdmfArrayType::Float64(), total_points);
        //     attr0->insert(0, data, total_points);
        //     // The heavy data set name is determined by the writer if not set
        //     m_grid->insert(attr0);

        //     // Attr1ibute
        //     boost::shared_ptr<XdmfAttribute> attr1 = XdmfAttribute::New();
        //     attr1->setName(name_ + "face1");
        //     attr1->setCenter(XdmfAttributeCenter::Node());
        //     attr1->setType(XdmfAttributeType::Vector());
        //     typename VecStorage::value_type data1[total_points];
        //     reindex_on_face<1>(storage_, m_local_cub, data1); // loops

        //     attr1->initialize(XdmfArrayType::Float64(), total_points);
        //     attr1->insert(0, data, total_points);
        //     // The heavy data set name is determined by the writer if not set
        //     m_grid->insert(attr1);

        //     // Attribute
        //     boost::shared_ptr<XdmfAttribute> attr2 = XdmfAttribute::New();
        //     attr2->setName(name_ + "face2");
        //     attr2->setCenter(XdmfAttributeCenter::Node());
        //     attr2->setType(XdmfAttributeType::Vector());
        //     typename VecStorage::value_type data2[total_points];
        //     reindex_on_face<2>(storage_, m_local_cub, data2); // loops

        //     attr2->initialize(XdmfArrayType::Float64(), total_points);
        //     attr2->insert(0, data, total_points);
        //     // The heavy data set name is determined by the writer if not set
        //     m_grid->insert(attr2);

        // }

        void set_information(std::string const& name_){
            // Information
            boost::shared_ptr<XdmfInformation>  i = XdmfInformation::New(); //# Arbitrary Name=Value Facility
            // i->setName("Time");
            i->setValue(name_);
            this->m_root->insert(i);  // XdmfDomain is the root of the tree

        }

        boost::shared_ptr< XdmfRectilinearGrid> m_grid;
    private:
        LocalCub const& m_local_cub;
    };


    // template <typename Storage, typename Cub, typename ... Int>
    // void print_element(std::ostream& ostream_, Storage const& storage_, Int ... indices){

    //     if(storage_.space_dim==4){
    //         assert(sizeof...(Int)==3);
    //         for(uint_t l=0; l<n_points; ++l)
    //             ostream_<<storage_(indices..., l)<<" ";
    //     }
    // }


}//namespace gridtools
