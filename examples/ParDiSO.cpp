#include "ParDiSO.hpp"
#include "CSRdouble.hpp"
#include "PardisoParameters.hpp"

#include <iostream>
using std::cout;

void ParDiSO::clear_(PardisoMemoryGroup memory_to_release)
{
  double ddum;
  int    idum;

  phase = int(memory_to_release);

  PARDISOCALL_D(pt,
                &maxfct,
                &mnum,
                &mtype,
                &phase,
                &idum,
                &ddum,
                &idum,
                &idum,
                &idum,
                &nrhs,
                &iparm[1],
                &msglvl,
                &ddum,
                &ddum,
                &error,
                &dparm[1]);
}



void ParDiSO::clear()
{
  clear_(PARDISO_MEMORY_FOR_FACTORS);
}

void ParDiSO::clearall()
{
  clear_(PARDISO_ALL_MEMORY);
}

// returns GB
double ParDiSO::memoryAllocated() const
{
  double peakMemorySymbolic = iparm[15];
  double permanentMemory = iparm[16] + iparm[17];

  // memory in kB
  double maxmem = peakMemorySymbolic > permanentMemory 
                ? peakMemorySymbolic : permanentMemory;

  return maxmem / 1024. / 1024.;
}

void ParDiSO::error_() const
{
  if (error != 0)
  {
    cout << "ParDiSO error: " << error << " --- ";
    switch (error)
    {
    case -1:
      cout << "input inconsistent\n";
      break;

    case -2:
      cout << "not enough memory\n";
      break;

    case -3:
      cout << "reordering problem\n";
      break;

    case -4:
      cout << "zero pivot, numerical factorization or iterative refinement problem\n";
      break;

    case -5:
      cout << "unclassified (internal) error\n";
      break;

    case -6:
      cout << "preordering failed (matrix type 11, 13 only)\n";
      break;

    case -7:
      cout << "diagonal matrix problem\n";
      break;

    case -8:
      cout << "32 bit integer overflow problem\n";
      break;

    case -10:
      cout << "No license file pardiso.lic found\n";
      break;

    case -11:
      cout << "License is expired.\n";
      break;

    case -12:
      cout << "Wrong username or hostname.\n";
      break;

    default:
      break;
    }
  }
}


ParDiSO::~ParDiSO()
{
  //clear_(PARDISO_ALL_MEMORY);
}

ParDiSO::ParDiSO()
{
  perm = 0;
}

ParDiSO::ParDiSO(int pardiso_mtype, int pardiso_msglvl, const char* filename)
{
  // --------------------------------------------------------------------
  // ..  Setup ParDiSO control parameters und initialize the solvers     
  //     internal adress pointers. This is only necessary for the FIRST  
  //     call of the ParDiSO solver.                                     
  // --------------------------------------------------------------------
  mtype  = pardiso_mtype;
  msglvl = pardiso_msglvl;
  msglvl = 0; // fabio msglvl hard coded;
  
  solver = 0;
  maxfct = 1;
  mnum   = 1;
  nrhs   = 1; 
  perm   = 0;

  PARDISOINIT_D(pt,  &mtype, &solver, &iparm[1], &dparm[1], &error);
  PardisoParameters params(filename);
  readParameters(&params); // reads only iparam[xxx]
  error_();
}



void ParDiSO::readParameters(PardisoParameters* params)
{
    iparm[1] = params->pardiso_iparm[1];
    iparm[2] = params->pardiso_iparm[2];
    iparm[3] = params->pardiso_iparm[3];
    iparm[4] = params->pardiso_iparm[4];
    iparm[5] = params->pardiso_iparm[5];
    iparm[8] = params->pardiso_iparm[8];
    iparm[10] = params->pardiso_iparm[10];
    iparm[11] = params->pardiso_iparm[11];
    iparm[12] = params->pardiso_iparm[12];
    iparm[13] = params->pardiso_iparm[13];
    iparm[18] = params->pardiso_iparm[18];
    iparm[19] = params->pardiso_iparm[19];
    iparm[21] = params->pardiso_iparm[21];
    iparm[28] = params->pardiso_iparm[28];
    iparm[29] = params->pardiso_iparm[29];
    iparm[32] = params->pardiso_iparm[32];

}


// @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
//
//                   D O U B L E    D R I V E R S
//
// @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

void ParDiSO::shiftIndices_(CSRdouble& A, int value)
{
  int i;
  for (i = 0; i <= A.nrows; i++)
  {
    A.pRows[i] += value;
  }

  for (i = 0; i < A.nonzeros; i++)
  {
    A.pCols[i] += value;
  }    
}

void ParDiSO::init(CSRdouble& A, int number_of_rhs)
{
  // --------------------------------------------------------------------
  // ..  Convert matrix from 0-based C-notation to Fortran 1-based       
  //     notation.                                                       
  // --------------------------------------------------------------------
  shiftIndices_(A, 1);


  // --------------------------------------------------------------------                                       
  //     Checks the consistency of the given matrix.                     
  //     Use this functionality only for debugging purposes              
  // --------------------------------------------------------------------
  //PARDISOCHECK_D(&mtype, &A.nrows, A.pData, A.pRows, A.pCols, &error);

  if (error != 0) 
  {
    printf("\nERROR in consistency of matrix: %d", error);
    exit(1);
  }

  // --------------------------------------------------------------------
  // ..  Reordering and Symbolic Factorization.  This step also allocates
  //     all memory that is necessary for the factorization.             
  // --------------------------------------------------------------------

  double ddum;
  nrhs = number_of_rhs;
  
  phase = 11;

  #ifdef DEBUG
  iparm[18] = -1; // report the number of nonzeros in the factors
  #endif

 
  PARDISOCALL_D(pt,
                &maxfct,
                &mnum,
                &mtype,
                &phase,
                &A.nrows,
                A.pData,
                A.pRows,
                A.pCols,
                perm,
                &nrhs,
                &iparm[1],
                &msglvl,
                &ddum,
                &ddum,
                &error,
                &dparm[1]);
  if (error != 0) 
  {
    printf("\nERROR in symbolic factorization of matrix: %d", error);
    exit(1);
  }

  #ifdef DEBUG_
  cout << "Number nonzeros in factors: " << iparm[18] << endl;
  #endif

  // --------------------------------------------------------------------
  // ..  Convert matrix from 1-based Fortran-notation to C 0-based       
  //     notation.                                                       
  // --------------------------------------------------------------------
  shiftIndices_(A, -1);
  error_();
}

void ParDiSO::factorize(CSRdouble& A)
{
  double ddum;

  // for factorization phase should be equal to 12
  phase = 22;

  shiftIndices_(A, 1);

  PARDISOCALL_D(pt,
                &maxfct,
                &mnum,
                &mtype,
                &phase,
                &A.nrows,
                A.pData,
                A.pRows,
                A.pCols,
                perm,
                &nrhs,
                &iparm[1],
                &msglvl,
                &ddum,
                &ddum,
                &error,
                &dparm[1]);

  shiftIndices_(A, -1);
  
  if (error != 0) 
  {
    printf("\nERROR in factorization of matrix: %d", error);
    exit(1);
  }

  #ifdef DEBUG_
  cout << "Total peak memory (kB) consumption is: " << memoryAllocated() << endl;
  #endif

  error_();
}

void ParDiSO::solve(CSRdouble& A, double* x, double* rhs, int number_of_rhs)
{
  // --------------------------------------------------------------------
  // ..  Back substitution and iterative refinement.                     
  // --------------------------------------------------------------------
  nrhs = number_of_rhs;
  phase = 33;

  shiftIndices_(A, 1);

  PARDISOCALL_D(pt,
                &maxfct,
                &mnum,
                &mtype,
                &phase,
                &A.nrows,
                A.pData,
                A.pRows,
                A.pCols,
                perm,
                &nrhs,
                &iparm[1],
                &msglvl,
                rhs,
                x,
                &error,
                &dparm[1]);

  shiftIndices_(A, -1);
  if (error != 0) 
  {
    printf("\nERROR in back-substitution of matrix: %d", error);
    exit(1);
  }
  error_();
}

//  A = H diagonal block, in
//  S = D local schur complement of single process, out
void ParDiSO::makeSchurComplement(CSRdouble& A, CSRdouble& S)
{
  double ddum;

  shiftIndices_(A, 1);
  //shiftIndices_(S, 1);
  

  // Check if this matrix is OK
  // PARDISOCHECK_D(&mtype, 
  //                &A.nrows, 
  //                A.pData, 
  //                A.pRows, 
  //                A.pCols, 
  //                &error);
 
  error_();
  phase     = 12;
  iparm[38] = S.nrows;


  // Perform symbolic analysis and numerical factorization
  PARDISOCALL_D(pt,
                &maxfct,
                &mnum,
                &mtype,
                &phase,
                &A.nrows,
                A.pData,
                A.pRows,
                A.pCols,
                perm,
                &nrhs, // needs to be initialized properly in order to call solve afterwards with multiple rhs
                &iparm[1],
                &msglvl,
                &ddum,
                &ddum,
                &error,
                &dparm[1]);


  S.nonzeros = int(iparm[39]);
  S.allocate(S.nrows, S.ncols, S.nonzeros);

  // calculate and store the Schur-complement
  PARDISOSCHUR_D(pt, 
                 &maxfct, 
                 &mnum, 
                 &mtype, 
                 S.pData, 
                 S.pRows, 
                 S.pCols);

  shiftIndices_(S, -1);
  shiftIndices_(A, -1);

  error_();
}
