#include "CopyStencil_h8gg0w8l.h"



extern "C"
{
    int run (uint_t dim1, uint_t dim2, uint_t dim3,void *out_data,void *in_data)
    {
        return !copystencil::test (dim1, dim2, dim3,out_data,in_data);
    }
}


int main()
{
    double* in= new double(512*512*60);
    double* out= new double(512*512*60);
    // double in[6*6*6];
    // double out[6*6*6];

    copystencil::test (512, 512, 60,out,in);
    delete in;
    delete out;
}
