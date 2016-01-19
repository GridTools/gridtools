#include <tools/io.hpp>
#include <stencil-composition/stencil-composition.hpp>

int main(){

    typedef gridtools::backend<gridtools::enumtype::Host, gridtools::enumtype::Naive> backend_t;

    // typedef backend_t
    //     ::storage_info<0, gridtools::layout_map<0,1,2> > meta_t;

    typedef backend_t
        ::storage_info<0, gridtools::layout_map<0,1,2,3,4> > meta_t;
    typedef typename backend_t
        ::storage_type<double, meta_t >::type storage_type;

    gridtools::uint_t ld1=2;
    gridtools::uint_t ld2=2;
    gridtools::uint_t ld3=2;

    gridtools::uint_t d1=20;
    gridtools::uint_t d2=20;
    gridtools::uint_t d3=20;
    gridtools::uint_t d4=ld1*ld2*ld3;


    // meta_t info(d1, d2, d3);

    meta_t info( d1, d2, d3 , d4, 3 );
    // storage_type storage_(info, [](gridtools::uint_t const& i, gridtools::uint_t const& j, gridtools::uint_t const& k){return (double)(i);});

    storage_type storage_(info, 0.);

    for (int i=0; i<d1; ++i)
        for (int j=0; j<d2; ++j)
            for (int k=0; k<d3; ++k)
            {
                for (int l=0; l<ld1; ++l)
                {
                    for (int m=0; m<ld2; ++m)
                    {
                        for (int n=0; n<ld3; ++n)
                        { //grid coordinates (missing division), low left corner in (-1,-1,-1)
                            // storage_(i,j,k,n*ld1*ld2+m*ld1+l)= (double) 2*(i*(ld1-1)+j*(ld2-1)+k*(ld3-1)+m+n+l)/((double) ((d1-1)*(ld1-1)+(d2-1)*(ld2-1)+(d3-1)*(ld3-1)+ld1-1+ld2-1+ld3-1))-1;
                            double x=(i*ld1+l)/((double)(d1-1)*ld1+(ld1-1))-.5;
                            double y=(j*ld2+m)/((double)(d2-1)*ld2+(ld2-1))-.5;
                            double z=(k*ld3+n)/((double)(d3-1)*ld3+(ld3-1))+1.;
                            storage_(i,j,k,n*ld1*ld2+m*ld1+l, 0)= (double) (x);
                            storage_(i,j,k,n*ld1*ld2+m*ld1+l, 1)= (double) (y);
                            storage_(i,j,k,n*ld1*ld2+m*ld1+l, 2)= (double) (z);
                        }
                    }
                }
            }

    typedef backend_t
        ::storage_info<0, gridtools::layout_map<0,1,2,3,4> > field_meta_t;
    typedef typename backend_t
        ::storage_type<double, field_meta_t >::type field_type;

    field_meta_t field_info( d1, d2, d3 , d4, 3 );

    field_type field_(field_info, 0.);

    typedef backend_t
        ::storage_info<0, gridtools::layout_map<0,1,2,3> > scalar_field_meta_t;
    typedef typename backend_t
        ::storage_type<double, scalar_field_meta_t >::type scalar_field_type;

    scalar_field_meta_t scalar_field_info( d1, d2, d3 , d4 );

    scalar_field_type scalar_field_(scalar_field_info, 0.);

    for (int i=0; i<d1; ++i)
        for (int j=0; j<d2; ++j)
            for (int k=0; k<d3; ++k)
            {
                for (int l=0; l<ld1; ++l)
                {
                    for (int m=0; m<ld2; ++m)
                    {
                        for (int n=0; n<ld3; ++n)
                        {   // field constant on the elements
                            // field_(i,j,k,n*ld1*ld2+m*ld1+l)= (double) i+j+k;
                            double x=(i*ld1+l)/((double)(d1-1)*ld1+(ld1-1))-.5;
                            double y=(j*ld2+m)/((double)(d2-1)*ld2+(ld2-1))-.5;
                            double z=(k*ld3+n)/((double)(d3-1)*ld3+(ld3-1))+1.;
                            //std::cout<<" x: "<<x<<" y: "<<y<<" z: "<<z<<std::endl;
                            double sph1= x*std::sqrt(1-(std::pow(y,2)/2.)-(std::pow(1., 2)/2.)+(std::pow(y*1., 2)/3.)) ;
                            double sph2= y*std::sqrt(1-(std::pow(x,2)/2.)-(std::pow(1., 2)/2.)+(std::pow(x*1., 2)/3.)) ;
                            double sph3= 1.*std::sqrt(1-(std::pow(y,2)/2.)-(std::pow(x, 2)/2.)+(std::pow(y*x, 2)/3.)) ;
                            field_(i,j,k,n*ld1*ld2+m*ld1+l,0)= sph1+sph1*z*0.1-x ;
                            field_(i,j,k,n*ld1*ld2+m*ld1+l,1)= sph2+sph2*z*0.1-y ;
                            field_(i,j,k,n*ld1*ld2+m*ld1+l,2)= sph3+sph3*z*0.1-z ;

                            scalar_field_(i,j,k,n*ld1*ld2+m*ld1+l)= z;
                        }
                    }
                }
            }

    // storage_.print();
    // for (int i=0; i<d1; ++i)
    //     std::cout<<" "<<storage_(i,0,0,0)<<" "<<storage_(i,0,0,1);
    // std::cout<<" x"<<std::endl;

    // for (int i=0; i<d2; ++i)
    //     std::cout<<" "<<storage_(0,i,0,0)<<" "<<storage_(0,i,0,ld2);
    // std::cout<<" x"<<std::endl;

    // for (int i=0; i<d3; ++i)
    //     std::cout<<" "<<storage_(0,0,i,0)<<" "<<storage_(0,0,i,ld1*ld2);
    // std::cout<<" x"<<std::endl;

    typedef backend_t::storage_info<0, gridtools::layout_map<0,1,2> > meta_local_t;
    meta_local_t meta_local_(ld1, ld2, ld3);

    gridtools::io_rectilinear<storage_type, meta_local_t> io_(storage_, meta_local_);

    // gridtools::io<storage_type, gridtools::enumtype::regular> io_(storage_);

    // storage_.print();


    io_.set_information("Time");
    io_.template set_attribute_vector<0>(field_, "test value");
    io_.template set_attribute_scalar<0>(scalar_field_, "z");
    io_.write("fuck");

    return 0;
}
