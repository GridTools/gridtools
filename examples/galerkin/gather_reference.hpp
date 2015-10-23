#pragma once

//reference for first order hexahedra
template <typename Storage>
void reference_1(Storage & out_, Storage const& in_, uint_t d1, uint_t d2, uint_t d3){

    //  z
    // /
    //  -x
    // |
    // y
    //
    //
    //   4----5
    //  /    /|
    // 0----1 |
    // |    | 7
    // |    |/
    // 2----3

    int order = 1;

    for (int i=0; i<d1; ++i)
        for (int j=0; j<d2; ++j)
            for (int k=1; k<d3; ++k)
                for(int ii=0;ii<8;++ii)
                    out_(i,j,k,ii)=(i+(j*10)+(k*100));;

    for (int i=1; i<d1; ++i)
        for (int j=1; j<d2; ++j)
            for (int k=1; k<d3; ++k)
            {

                out_(i,j,k,0) += in_(i-1,j,k,4)+in_(i,j,k,0);
                out_(i,j,k,1) += in_(i-1,j,k,5)+in_(i,j,k,1);
                out_(i,j,k,2) += in_(i-1,j,k,6)+in_(i,j,k,2);
                out_(i,j,k,3) += in_(i-1,j,k,7)+in_(i,j,k,3);

                out_(i,j,k,0) += in_(i,j-1,k,2)+in_(i,j,k,0);
                out_(i,j,k,1) += in_(i,j-1,k,3)+in_(i,j,k,1);
                out_(i,j,k,4) += in_(i,j-1,k,6)+in_(i,j,k,4);
                out_(i,j,k,5) += in_(i,j-1,k,7)+in_(i,j,k,5);

                out_(i,j,k,0) += in_(i,j,k-1,1)+in_(i,j,k,0);
                out_(i,j,k,2) += in_(i,j,k-1,3)+in_(i,j,k,2);
                out_(i,j,k,4) += in_(i,j,k-1,5)+in_(i,j,k,4);
                out_(i,j,k,6) += in_(i,j,k-1,7)+in_(i,j,k,6);

}

}
