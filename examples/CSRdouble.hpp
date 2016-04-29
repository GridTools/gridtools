#ifndef CSRdouble_hpp
#define CSRdouble_hpp

#include <string>
#include <vector>
#include <fstream>
#include <iomanip>

// using std::complex;
// using std::endl;
// using std::setw;
// using std::cout;
// using std::fstream;
using std::ios;
using std::string;
using std::vector;

enum MatrixStorage
{
  NOT_SET   = 0,
  NORMAL    = 1,
  TRANSPOSE = 2,
  SYMMETRIC = 3,  
  HERMITIAN = 4,  
};

class CSRdouble {
public:
    string         name;
    int            nrows;
    int            ncols;
    int            nonzeros;
    int*           pRows;
    int*           pCols;
    double*        pData;
    MatrixStorage  matrixType;

public:
    CSRdouble();
    CSRdouble(int nrows_, int ncols_, int nonzeros_, const int* pRows_, const int* pCols_, const double* pData_);
    ~CSRdouble();

    void  allocate ( int n, int m, int nzeros );
    void  loadFromFile ( const char* file, bool hasCols, ios::openmode mode = ios::out );
    void  getBlock(CSRdouble& A, const int n, const int bid);

    void  loadFromFileCOO ( const char* file );
    void  make ( int n, int m, int nzeros, int* prows, int* pcols, double* pdata );
    void  make (int n, int m, vector<vector<int> >& vvcols, vector<vector<double> >& vvdata);
    void  make2 ( int n, int m, int nzeros, int* prows, int* pcols, double* pdata );
    void  transposeIt ( int block_size );

    void  addBCSR ( CSRdouble& B );
    void  extendrows ( CSRdouble& B, int startrowB, int nrowsB );

    void  residual ( double* r, double* x, double* b );
    void  multiply ( double* x, double* y );
    void  multiplyS ( double* x, double* y );
    void  multiplyN ( double* x, double* y );
    void  multiplyT ( double* x, double* y );
    void  alphaAx_plus_betay(double alpha, double beta, double* x, double* y);
    void  sortColumns();
    void  fillSymmetric();
    void  fillSymmetricNew();
    void  reduceSymmetric();
    void  writeToFile ( const char* filename, ios::openmode mode = ios::out ) const;
    void  savedebug ( const char* filename ) const;
    double memoryAllocated() const;
    void sendTo(int rank, int flag);
    int receive(int rank);
    void  clear();
};


#endif
