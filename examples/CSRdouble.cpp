#include "CSRdouble.hpp"
#include "searchingsorting.hpp"

#include <mpi.h>
#include <cstdlib>
#include <cmath>
#include <cassert>
#include <cstring>
#include <climits>

#ifdef DEBUG
#include "mpi.h"
#endif


using std::endl;
using std::setw;
using std::cout;
using std::cerr;
using std::fstream;
using std::vector;

CSRdouble::~CSRdouble()
{
    clear();
}

CSRdouble::CSRdouble()
{
    nrows          = 0;
    ncols          = 0;
    nonzeros       = 0;
    pRows          = NULL;
    pCols          = NULL;
    pData          = NULL;
    name           = "UnNamed";
}

CSRdouble::CSRdouble(int nrows_, int ncols_, int nonzeros_, const int* pRows_, const int* pCols_, const double* pData_)
{
    allocate(nrows_, ncols_, nonzeros_);
    for (int i = 0; i < nrows+1; i++)
    {
        pRows[i] = pRows_[i];
    }
    for (int i = 0; i < nonzeros; i++)
    {
        pCols[i] = pCols_[i];
        pData[i] = pData_[i];
    }

    name           = "UnNamed";

    //correct Fortran one-based indexing into C zero-based
    int i0 = pRows[0];
    for (int i = 0; i < nrows+1; i++)
    {
        pRows[i] -= i0;
    }

    for (int i = 0; i < nonzeros; i++)
    {
        pCols[i] -= i0;
    }
}

void CSRdouble::clear()
{
    #ifdef DEBUG_memory
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    cout << "Clearing " << memoryAllocated() << " GB at rank #" << rank << endl;
    #endif

    if(pData != NULL)
        delete [] pData;
    pData = NULL;
    if(pRows != NULL)
        delete [] pRows;
    pRows = NULL;
    if(pCols != NULL)
        delete [] pCols;
    pCols = NULL;

    nrows          = 0;
    ncols          = 0;
    nonzeros       = 0;
}

void CSRdouble::allocate(int n, int m, int nzeros)
{
    nrows            = n;
    ncols            = m;
    nonzeros         = nzeros;

    #ifdef DEBUG_memory
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    cout << "Allocating " << memoryAllocated() << " GB at rank #" << rank << endl;
    #endif

    assert(nonzeros < INT_MAX && nonzeros > 0);
    assert(nrows > 0);
    assert(ncols > 0);

    try
    {
        pRows            = new int[nrows + 1];
        pCols            = new int[nonzeros];
        pData            = new double[nonzeros];
    }
    catch (const std::bad_alloc& err) 
    {
        cout << "Error in allocation at CSRdouble::allocate" << std::endl;
        cout << "Need " << memoryAllocated() << " GB" << std::endl;
        cout << err.what() << std::endl;

        #ifdef DEBUG
        int rank;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        cout << "Error at rank #" << rank << endl;
        #endif
    }
}

void CSRdouble::makePreconditioner(int ni, int nj, int nk)
{
    //size of the preconditioner matrix
    int n = ni*nj*nk;

    //nonzeros in local preconditioner
    int nnz = ni*nj*(nk-1)*2 + (ni*nj + 4*2 + (ni-2)*3*2 + (nj-2)*3*2 + (ni-2)*(nj-2)*4)*nk;

    allocate(n, n, nnz);

    cout << "Construction of M with size " << ni << "x" << nj << "x" << nk << " with " << nnz << " non-zeros" << endl;

    auto index = [ni,nj] (int i, int j, int k) { return k*nj*ni + j*ni + i; };
    pRows[0] = 0;
    int nnz_counter = 0;

    for (int k = 0; k < nk; k++)
        for (int j = 0; j < nj; j++)
            for (int i = 0; i < ni; i++)
            {
                int nelem = 0; //number of elements in the row of matrix

                //down neighbor
                if (k-1 >= 0)
                {
                    pCols[nnz_counter] = index(i,j,k-1);
                    pData[nnz_counter] = -1;
                    nelem++;
                    nnz_counter++;   
                };

                //south neighbor
                if (j-1 >= 0)
                {
                    pCols[nnz_counter] = index(i,j-1,k);
                    pData[nnz_counter] = -1;
                    nelem++;
                    nnz_counter++;   
                };

                //west neighbor
                if (i-1 >= 0)
                {
                    pCols[nnz_counter] = index(i-1,j,k);
                    pData[nnz_counter] = -1;
                    nelem++;
                    nnz_counter++;   
                };

                //diagonal element
                pCols[nnz_counter] = index(i,j,k);
                pData[nnz_counter] = 6;
                nelem++;
                nnz_counter++;

                //east neighbor
                if (i+1 < ni)
                {
                    pCols[nnz_counter] = index(i+1,j,k);
                    pData[nnz_counter] = -1;
                    nelem++;
                    nnz_counter++;   
                };

                //north neighbor
                if (j+1 < nj)
                {
                    pCols[nnz_counter] = index(i,j+1,k);
                    pData[nnz_counter] = -1;
                    nelem++;
                    nnz_counter++;   
                };

                //up neighbor
                if (k+1 < nk)
                {
                    pCols[nnz_counter] = index(i,j,k+1);
                    pData[nnz_counter] = -1;
                    nelem++;
                    nnz_counter++;   
                };


                //mark position where next CSR row starts
                pRows[index(i,j,k)+1] = pRows[index(i,j,k)] + nelem;
            }

    assert(nnz_counter == nnz);

    writeToFile("M.csr");
}

void CSRdouble::make(int n, int m, int nzeros, int* prows,
                     int* pcols, double* pdata)
{
    // this is used to set the sparse structure mainly; here the pointer
    // to the values, pdata, is not necessarily ready (initialized)
    nrows            = n;
    ncols            = m;
    nonzeros         = nzeros;
    pRows            = prows;
    pCols            = pcols;
    pData            = pdata;
    name             = "UnNamed";
}

void  CSRdouble::make(int n, int m, vector<vector<int> >& vvcols, vector<vector<double> >& vvdata)
{
    nonzeros = 0;
    for (int i = 0; i < n; i++)
    {
        nonzeros += vvcols[i].size();
    }

    allocate(n, m, nonzeros);
    pRows[0] = 0;

    int index = 0;
    for (int i = 0; i < n; i++)
    {
        for (size_t j = 0; j < vvcols[i].size(); j++)
        {
            pCols[index] = vvcols[i][j];
            pData[index] = vvdata[i][j];

            index++;
        }
        pRows[i+1] = pRows[i] + vvcols[i].size();
    }

    sortColumns();
}

void CSRdouble::make2(int n, int m, int nzeros, int* prows,
                      int* pcols, double* pdata)
{
    // this is used to set the sparse structure mainly; here the pointer
    // to the values, pdata, is not necessarily ready (initialized)
    nrows            = n;
    ncols            = m;
    nonzeros         = nzeros;

    allocate(nrows, ncols, nonzeros);

    memcpy(pRows,prows,(n+1) * sizeof(int));
    memcpy(pCols,pcols,nzeros * sizeof(int));
    memcpy(pData,pdata,nzeros * sizeof(double));
    name             = "UnNamed";
}

void CSRdouble::sortColumns()
{
    for (int i = 0; i < nrows; i++)
    {
        int index     = pRows[i];
        int entries   = pRows[i+1] - pRows[i];
        int* pcols    = pCols + index;
        double* pdata = pData + index;

        heapsort(entries, pcols, pdata);
    }
}

// r = Ax - b
void CSRdouble::residual(double* r, double* x, double* b)
{
    multiply(x, r);

    for (int i = 0; i < nrows; i++)
    {
        r[i] = r[i] - b[i];
    }
}

void CSRdouble::transposeIt(int block_size)
{
    // these are standard for the transpose of a matrix
    int transpose_nrows     = ncols;
    int transpose_ncols     = nrows;
    int transpose_nonzeros  = nonzeros;

    int* transpose_prows;
    int* transpose_pcols;
    double* transpose_pdata;

    try
    {
        transpose_prows    = new int[transpose_nrows + 1];
        transpose_pcols    = new int[transpose_nonzeros];
        transpose_pdata = new double[transpose_nonzeros*block_size];
    }
    catch (const std::bad_alloc& err)
    {
        cout << "Error in allocation at CSRdouble::transposeIt" << std::endl;
        cout << err.what() << std::endl;
    }

    int* rowptr             = transpose_pcols;
    double* dataptr         = transpose_pdata;

    // now lets create the CSR structure of the transpose
    int i, j, index, from, to;
    vector<vector<int> > transpose_veccols(transpose_nrows);
    vector<vector<double> > transpose_vecdata(transpose_nrows);

    for (i = 0; i < nrows; i++)
    {
        from = pRows[i];
        to = pRows[i+1];
        for (index = from; index < to; index++)
        {
            j = pCols[index];
            transpose_veccols[j].push_back(i);
            vector<double>& v = transpose_vecdata[j];

            for (int k = 0; k < block_size; k++)
            {
                v.push_back(pData[index*block_size + k]);
            }
        }
    }

    // we almost have our sparse structure constructed now;
    // all what is left is to copy it from the vector.
    transpose_prows[0] = 0;
    for (i = 0; i < transpose_nrows; i++)
    {
        int entries = int(transpose_veccols[i].size());
        memcpy(rowptr,  &transpose_veccols[i].front(), entries*sizeof(int));
        memcpy(dataptr, &transpose_vecdata[i].front(), entries*block_size*sizeof(double));
        rowptr  += entries;
        dataptr += entries*block_size;

        transpose_prows[i+1] = transpose_prows[i] + entries;
    }

    nrows          = transpose_nrows;
    ncols          = transpose_ncols;
    nonzeros       = transpose_nonzeros;
    pData          = transpose_pdata;
    pCols          = transpose_pcols;
    pRows          = transpose_prows;

    name           = "Transpose of an UnNamed";
    matrixType     = NORMAL;
}

void CSRdouble::multiply(double* x, double* y)
{
    switch (matrixType)
    {
    case NORMAL:
        multiplyN(x, y);
        break;

    case TRANSPOSE:
        multiplyT(x, y);
        break;

    case SYMMETRIC:
        multiplyS(x, y);
        break;

    default:
        cout << "matrix: \'" << name << "\' multiply, matrixType not set" << endl;
        break;
    }
}

void CSRdouble::multiplyS(double* x, double* b)
{
    memset(b, 0, nrows*sizeof(double));
    for (int i = 0; i < nrows; i++)
    {
        double x_i  = x[i];
        double a_ii = pData[pRows[i]];
        double sum  = a_ii*x_i;

        for (int index = pRows[i]+1; index < pRows[i+1]; index++)
        {
            int j       = pCols[index];
            double a_ij = pData[index];

            sum        += a_ij*x[j];
            b[j]       += a_ij*x[i];
        }
        b[i]         += sum;
    }
}

void CSRdouble::multiplyN(double* x, double* y)
{
    for (int i = 0; i < nrows; i++)
    {
        double sum = 0.0;
        for (int index = pRows[i]; index < pRows[i+1]; index++)
        {
            int j = pCols[index];
            sum += pData[index] * x[j];
        }

        y[i] = sum;
    }
}

void CSRdouble::multiplyT(double* x, double* y)
{
    memset(y, 0, ncols*sizeof(double));
    for (int i = 0; i < nrows; i++)
    {
        for (int index = pRows[i]; index < pRows[i+1]; index++)
        {
            int j = pCols[index];
            y[j] += pData[index] * x[i];
        }
    }
}

// y = aAx + by
void CSRdouble::alphaAx_plus_betay(double alpha, double beta, double* x, double* y)
{
    // r = A*x
    int n;
    if (matrixType == TRANSPOSE)
        n = ncols;
    else
        n = nrows;

    vector<double> r(n);
    multiply(x, &r[0]);

    for (int i = 0; i < n; i++)
    {
        y[i] = alpha*r[i] + beta*y[i];
    }
}

void CSRdouble::writeToFile(const char* filename, ios::openmode mode) const
{
    cout << "\t---> Dumping matrix to file: " << filename << endl;

    fstream fout(filename, ios::out | mode);
    if (!fout.is_open())
    {
        cout << "could not open file " << filename << " for output\n";
        return;
    }

    if (mode == ios::binary)
    {
        fout.seekp(0);
        fout.write((char*)&nrows, sizeof(int));

        fout.seekp(sizeof(int));
        fout.write((char*)&ncols, sizeof(int));

        fout.seekp(sizeof(int)*2);
        fout.write((char*)&nonzeros, sizeof(int));

        fout.seekp(sizeof(int)*3);
        fout.write((const char*)pRows, sizeof(int)*(nrows+1));

        fout.seekp(sizeof(int)*(nrows+1+3));
        fout.write((const char*)pCols, sizeof(int)*nonzeros);

        fout.seekp(sizeof(int)*(nrows+1+3 + nonzeros));
        fout.write((const char*)pData, sizeof(double)*nonzeros);
        fout.close();
    }
    else
    {
        fout << nrows << "\n";
        fout << ncols << "\n";
        fout << nonzeros << "\n";

        #ifdef VERBOSE
        cout << "nrows: " << nrows << endl;
        cout << "ncols: " << ncols << endl;
        cout << "nonzeros: " << nonzeros << endl;
        #endif

        int i;
        for (i = 0; i < nrows+1; i++)
        {
            fout << pRows[i] << "\n";
        }

        for (i = 0; i < nonzeros; i++)
        {
            fout << pCols[i] << "\n";
        }

        fout.setf(ios::scientific, ios::floatfield);
        fout.precision(16);

        for (i = 0; i < nonzeros; i++)
        {
            fout << pData[i] << "\n";
        }
    }

    fout.close();
}

void CSRdouble::loadFromFile(const char* file, bool hasCols, ios::openmode mode)
{
    fstream fin(file, ios::in | mode);

    #ifdef DEBUG
    cout << "opening file: " << file << " in mode: ";
    #endif

    if (!fin.is_open())
    {
        cout << "Couldn't open file ... " << file << "\n";
        exit(1);
    }

    if (mode == ios::binary)
    {
        #ifdef DEBUG
        cout << " binary" << std::endl;
        #endif

        fin.seekg(0);
        fin.read((char*)&nrows, sizeof(int));

        if (hasCols)
        {
            fin.seekg(sizeof(int));
            fin.read((char*)&ncols, sizeof(int));
        }
        else
        {
            ncols = nrows;
        }

        fin.seekg(sizeof(int)*2);
        fin.read((char*)&nonzeros, sizeof(int));

        #ifdef DEBUG
        cout << "nrows:    " << nrows    << "\n";
        cout << "ncols:    " << ncols    << "\n";
        cout << "nonzeros: " << nonzeros << "\n";
        #endif

        allocate(nrows, ncols, nonzeros);

        fin.seekg(sizeof(int)*3);
        fin.read((char*)pRows, sizeof(int)*(nrows+1));

        fin.seekg(sizeof(int)*(nrows+1+3));
        fin.read((char*)pCols, sizeof(int)*nonzeros);

        fin.seekg(sizeof(int)*(nrows+1+3 + nonzeros));
        fin.read((char*)pData, sizeof(double)*nonzeros);

    }
    else
    {
        #ifdef DEBUG
        cout << " ascii" << std::endl;
        #endif

        fin >> nrows;
        if (hasCols)
        {
            fin >> ncols;
        }
        else
        {
            ncols = nrows;
        }
        fin >> nonzeros;

        #ifdef DEBUG
        cout << "nrows:    " << nrows    << std::endl;
        cout << "ncols:    " << ncols    << std::endl;
        cout << "nonzeros: " << nonzeros << std::endl;
        #endif
        
        allocate(nrows, ncols, nonzeros);

        int i;
        for (i = 0; i < nrows+1; i++)
        {
            fin >> pRows[i];
        }

        for (i = 0; i < nonzeros; i++)
        {
            fin >> pCols[i];
        }

        for (i = 0; i < nonzeros; i++)
        {
            fin >> pData[i];
        }

    }
    fin.close();

    //correct Fortran one-based indexing into C zero-based
    int i0 = pRows[0];
    for (int i = 0; i < nrows+1; i++)
    {
        pRows[i] -= i0;
    }

    for (int i = 0; i < nonzeros; i++)
    {
        pCols[i] -= i0;
    }
}

void CSRdouble::loadFromFileCOO(const char* file)
{
    fstream fin(file, ios::in);
    cout << "opening file: " << file << " in mode: ";
    if (!fin.is_open())
    {
        cout << "Couldn't open file ... " << file << "\n";
        exit(1);
    }


    cout << " ascii" << std::endl;

    fin >> nrows >> ncols >> nonzeros;

    cout << "nrows:    " << nrows    << std::endl;
    cout << "ncols:    " << ncols    << std::endl;
    cout << "nonzeros: " << nonzeros << std::endl;

    allocate(nrows, ncols, nonzeros);

    int i, j, i0;
    double aij;
    int index;
    vector<vector<int> > vvcols(nrows+1);
    vector<vector<double> > vvdata(nrows+1);
    for (index = 0; index < nonzeros; index++)
    {
        fin >> i >> j >> aij;
        vvcols[i].push_back(j);
        vvdata[i].push_back(aij);
    }

    if (vvcols[0].empty())
        i0 = 1;
    else
        i0 = 0;

    fin.close();

    index = 0;
    pRows[0] = 0;
    for (i = i0; i < nrows+i0; i++)
    {
        int entries = vvcols[i].size();
        heapsort(entries, &vvcols[i][0], &vvdata[i][0]);

        memcpy(&pData[index], &vvdata[i][0], entries*sizeof(double));
        memcpy(&pCols[index], &vvcols[i][0], entries*sizeof(int));

        index += entries;
        pRows[i+1-i0] = index;
    }



    for (i = 0; i < nrows+1; i++)
    {
        pRows[i] -= i0;
    }

    for (i = 0; i < nonzeros; i++)
    {
        pCols[i] -= i0;
    }
}

// This method fills the symmetric sparse structure
// so that the matrix is not any more in upper or lower
// triangular form.
void CSRdouble::fillSymmetric()
{
    int nonzeros;
    int  n        = this->nrows;
    int* prows    = this->pRows;
    int* pcols    = this->pCols;
    double* pdata = this->pData;

    vector<vector<double> > vA(n);
    vector<vector<int> >    vcols(n);

    int i;
    for (i = 0; i < n; i++)
    {
        for (int index = prows[i]; index < prows[i+1]; index++)
        {
            int j = pcols[index];

            vcols[i].push_back(j);
            double a_ij = pdata[index];
            vA[i].push_back(a_ij);

            // this is the j column in the i-th row; now we need to find the
            // i-th column in the j-th row; If it is there we do nothing; if
            // not then we need to add it
            if (i != j)
            {
                bool found = false;
                for (int k = prows[j]; k < prows[j+1]; k++)
                {
                    int col = pcols[k];
                    if (col == i)
                    {
                        found = true;
                        break;
                    }
                }

                if ( !found )
                {
                    //cout << "The matrix is not Structurally Symmetric\n";
                    vcols[j].push_back(i);
                    vA[j].push_back(a_ij);
                }
            }
        }
    }

    int* ia;
    try
    {
        ia = new int[n+1];
    }
    catch (const std::bad_alloc& err)
    {
        cout << "Error in allocation at CSRdouble::fillSymmetric" << std::endl;
        cout << err.what() << std::endl;
    }

    ia[0]   = 0;
    for (i = 0; i < n; i++)
    {
        ia[i+1] = ia[i] + vcols[i].size();
    }

    nonzeros   = ia[n];

    int* ja;
    double* a;

    try
    {
        ja    = new int[nonzeros];
        a  = new double[nonzeros];
    }
    catch (const std::bad_alloc& err)
    {
        cout << "Error in allocation at CSRdouble::fillSymmetric" << std::endl;
        cout << err.what() << std::endl;
    }

    for (i = 0; i < n; i++)
    {
        int index = ia[i];
        int entries = vcols[i].size();
        for (int j = 0; j < entries; j++)
        {
            ja[index + j] = vcols[i][j];
            a[index + j]  = vA[i][j];
        }

        if (entries > 1)
            heapsort(entries, &ja[index], &a[index]);
    }

    delete[] pRows;
    delete[] pCols;
    delete[] pData;

    make(n, n, nonzeros, ia, ja, a);
    matrixType = NORMAL;
}

// This method deletes the symmetric sparse structure
// so that the matrix is stored in upper
// triangular form.
void CSRdouble::reduceSymmetric()
{
    int nonzeroes, nnz_count;
    int  n = nrows;
    int* prows    ;
    int* pcols    ;
    double* pdata ;

    vector<vector<double> > vA(n);
    vector<vector<int> >    vcols(n);
    nonzeroes = (nonzeros + nrows)/2;

    try
    {
        prows = new int[n+1];
        pcols = new int[nonzeroes];
        pdata = new double[nonzeroes];
    }
    catch (const std::bad_alloc& err)
    {
        cout << "Error in allocation at CSRdouble::reduceSymmetric" << std::endl;
        cout << err.what() << std::endl;
    }

    nnz_count=0;
    prows[0]=0;
    for (int i = 0; i < n; i++)
    {
        for (int index = pRows[i]; index < pRows[i+1]; index++)
        {
            int j = pCols[index];
            if(j>=i) {
                pcols[nnz_count]=j;
                pdata[nnz_count]=pData[index];
                ++nnz_count;
            }
        }
        prows[i+1]=nnz_count;
    }

    if (nnz_count != nonzeroes)
        cout << "Nonzeroes do not match, nonzero_counter= " << nnz_count << "; nonzeroes= " << nonzeroes <<endl;

    delete[] pRows;
    delete[] pCols;
    delete[] pData;

    make(n, n, nonzeroes, prows, pcols, pdata);
    matrixType = SYMMETRIC;
}

void CSRdouble::savedebug(const char* filename) const
{
    fstream fout(filename, ios::out);

    int i, index, j;

    fout.setf(ios::scientific, ios::floatfield);
    fout.precision(16);

    for (i = 0; i < nrows; i++)
    {
        fout << "row #" << i << "\n";
        fout << "============\n";

        for (index = pRows[i]; index < pRows[i+1]; index++)
        {
            j = pCols[index];
            fout << setw(12) << j << setw(1) << ":" << setw(25) << pData[index] << "\n";
        }
        fout << "\n";
    }
}

void CSRdouble::fillSymmetricNew()
{
  vector<int> nnz(nrows, 0);

    int  n        = this->nrows;
    nonzeros   = 2*nonzeros - n;

    int* ia;
    int* ja;
    double* a;

    try{
        ia    = new int[n+1];
        ja    = new int[nonzeros];
        a     = new double[nonzeros];
        ia[0] = 0;
    }
        catch (const std::bad_alloc& err) {
        cout << "Error in allocation at CSRdouble::fillSymmetricNew" << std::endl;
        cout << err.what() << std::endl;
    }


    for (int i = 0; i < n; i++)
    {
        for (int index = pRows[i]; index < pRows[i+1]; index++)
        {
            int j       = pCols[index];
            nnz[i]     += 1;
            if (i != j)
               nnz[j]  += 1;
        }
    }

    for (int i = 0; i < n; i++)
    {
      ia[i+1] = ia[i] + nnz[i];
    }

    // now ia is formed and we need to form ja and a
    nnz.assign(n, 0);
    for (int i = 0; i < n; i++)
    {
        for (int index = pRows[i]; index < pRows[i+1]; index++)
        {
            int j       = pCols[index];
            if (i != j)
            {
              ja[ ia[j] + nnz[j] ] = i;
               a[ ia[j] + nnz[j] ] = pData[index];
               nnz[j]             += 1;
            }

            ja[ ia[i] + nnz[i] ] = j;
             a[ ia[i] + nnz[i] ] = pData[index];
             nnz[i]             += 1;
        }
    }

    // destroy the old pointers
    delete[] pRows;
    delete[] pCols;
    delete[] pData;

    // and build a new matrix out of ia,ja,a
    make(n, n, nonzeros, ia, ja, a);
    matrixType = NORMAL;
}

inline void mpi_check(int mpi_call)
{
    if ((mpi_call) != 0) { 
        cerr << "MPI Error detected!" << endl;
        exit(1);
    }
}

// Sends CSR object to MPI process with given rank
void CSRdouble::sendTo(int rank, int tag)
{
    int CSRheader[4];
    CSRheader[0] = nrows;
    CSRheader[1] = ncols;
    CSRheader[2] = nonzeros;
    CSRheader[3] = tag;
    mpi_check(MPI_Send(CSRheader, 4, MPI_INT, rank, 0, MPI_COMM_WORLD));

    mpi_check(MPI_Send(pRows, nrows + 1, MPI_INT, rank, 1+tag, MPI_COMM_WORLD)); 
    mpi_check(MPI_Send(pCols, nonzeros, MPI_INT, rank, 2+tag, MPI_COMM_WORLD));
    mpi_check(MPI_Send(pData, nonzeros, MPI_DOUBLE, rank, 3+tag, MPI_COMM_WORLD));
}

// Receives CSR object from the process rank
// returns tag that is used as partition ID in our application
int CSRdouble::receive(int rank)
{
    int CSRheader[4];

    // receive CSR object
    MPI_Status status;
    int count;
    mpi_check(MPI_Recv(CSRheader, 4, MPI_INT, rank, 0, MPI_COMM_WORLD, &status));
    MPI_Get_count(&status, MPI_INT, &count);
    assert(4 == count);

    nrows = CSRheader[0];
    ncols = CSRheader[1];
    nonzeros = CSRheader[2];

    // partition id in global terms
    int tag = CSRheader[3];

    allocate(nrows, ncols, nonzeros);

    mpi_check(MPI_Recv(pRows, nrows + 1, MPI_INT, rank, 1+tag, MPI_COMM_WORLD, &status));
    MPI_Get_count(&status, MPI_INT, &count);
    assert(nrows+1 == count);

    mpi_check(MPI_Recv(pCols, nonzeros, MPI_INT, rank, 2+tag, MPI_COMM_WORLD, &status));
    MPI_Get_count(&status, MPI_INT, &count);
    assert(nonzeros == count);

    mpi_check(MPI_Recv(pData, nonzeros, MPI_DOUBLE, rank, 3+tag, MPI_COMM_WORLD, &status));
    MPI_Get_count(&status, MPI_DOUBLE, &count);
    assert(nonzeros == count);

    return tag;
}

// return memory usage in GB
double CSRdouble::memoryAllocated() const
{
    double memory = 0.;

    if (nrows != 0)
        memory = 8.0 * nonzeros + 4 * nonzeros + 4 * (nrows+1);

    return memory / 1024. / 1024. / 1024.;
    
}
