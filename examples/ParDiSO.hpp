#ifndef ParDiSO_hpp
#define ParDiSO_hpp


#ifdef AIX
#define PARDISOINIT pardisoinit
#define PARDISOCALL pardiso
#define PARDISOSCHUR pardiso_get_schur
#define PARDISOCHECK pardiso_chkmatrix
#elif WIN32
#define PARDISOINIT PARDISOINIT
#define PARDISOCALL PARDISO
#define PARDISOSCHUR PARDISO_GET_SCHUR
#define PARDISOCHECK PARDISO_CHKMATRIX
#else
#define PARDISOINIT_D  pardisoinit_d_
#define PARDISOCALL_D  pardiso_d_
#define PARDISOSCHUR_D pardiso_get_schur_d_
#define PARDISOCHECK_D pardiso_chkmatrix_d_
#define PARDISOINIT_Z  pardisoinit_z_
#define PARDISOCALL_Z  pardiso_z_
#define PARDISOSCHUR_Z pardiso_get_schur_z_
#define PARDISOCHECK_Z pardiso_chkmatrix_z_
#endif

enum PardisoMode 
{
  SOLVE_SYSTEM       = 1,
  SCHUR_COMPLEMENT   = 2,
  ENTRIES_OF_INVERSE = 3
};

#include "CSRdouble.hpp"
#include "PardisoParameters.hpp"

// ===== forward declarations =====
extern  "C" int PARDISOCHECK_D(int*, int*, double*, int*, int*, int*);

extern  "C" int PARDISOINIT_D(void *, int *, int *, int *, double *, int *);
extern  "C" int PARDISOCALL_D(void *, int *, int *, int *, int *, int *,
			      double *, int *, int *, int *, int *, int *,
			      int *, double *, double *, int *, double *);

extern  "C" int PARDISOSCHUR_D(void*, int*, int*, int*, double*, int*, int*);


// ================================
enum PardisoMemoryGroup
{
  PARDISO_ALL_MEMORY         = -1,
  PARDISO_MEMORY_FOR_FACTORS =  0,
};



class ParDiSO
{
public:
// PT
// -------------------------------------------------------------------
// Internal solver memory pointer pt, 
// 32-bit: int pt[64]; 
// 64-bit: long int pt[64]         
// or void *pt[64] should be OK on both architectures  
// -------------------------------------------------------------------
  void* pt[64];


// IPARM
// -------------------------------------------------------------------
// On entry: IPARM is an integer array of size 64 that is used to pass
// various parameters to PARDISO and to return some useful information
// after the execution of the solver. PARDISOINIT fills IPARM(1),
// IPARM(2), and IPARM(4) through IPARM(64) with default values and
// uses them.  See section 2.3 for a detailed description.  Note: Note
// that there is no default value for IPARM(3), which reflecsts the
// number of processors and this value must always be supplied by the
// user

// IPARM(1) Use default options.
// ---------------------------------------------------------
// 0* Set all entries to their default values except IPARM(3)
// ---------------------------------------------------------


// IPARM(2) Use METIS reordering.
// -----------------------------------------
// 0  Do not use METIS.
// 2* Use METIS nested dissection reordering
// -----------------------------------------


// IPARM(3) Number of processors.
// ----------------------------------------------------------
// p Number of OPENMP threads. This must be identical to the
//   environment variable OMP NUM THREADS.
// ----------------------------------------------------------


// IPARM(4) Do preconditioned CGS iterations. Default is 0.
// -------------------------------------------------------------------

// On entry: This parameter controls preconditioned CGS [12] for
// nonsymmetric or structural symmetric matrices and
// Conjugate-Gradients for symmetric matrices. IPARM(4) has the form
// IPARM(4) = 10 * L + K.

// #################
// # K Description #
// #################
// 0 The factorization is always computed as required by PHASE.
// 1 CGS iteration replaces the computation of LU.  The preconditioner
//   is LU that was computed at a previous step (the first step or last
//   step with a failure) in a sequence of solutions needed for identical
//   sparsity patterns.
// 2 CG iteration for symmetric matrices replaces the computation of
//   LU.  The preconditioner is LU that was computed at a previous step
//   (the first step or last step with a failure) in a sequence of
//   solutions needed for identical sparsity patterns.

// #################
// # L Description #
// #################
// The value L controls the stopping criterion of the Krylov-Subspace
// iteration: error_CGS = 10^{-L} is used in the stopping criterion
// ||dx_i||/||dx_1|| < error_CGS with ||dx_i|| = ||(LU)^{-1}r_i|| and
// r_i is the residuum at iteration i of the preconditioned
// Krylov-Subspace iteration.

// ############
// # Strategy #
// ############
// A maximum number of 150 iterations is fixed by expecting that the
// iteration will converge before consuming half the factorization
// time. Intermediate convergence rates and residuum excursions are
// checked and can terminate the iteration process. If PHASE=23, then
// the factorization for a given A is automatically recomputed in these
// caseswhere the Krylov-Subspace iteration failed and the
// corresponding direct solution is returned. Otherwise the solution
// from the preconditioned Krylov Subspace iteration is returned. Using
// PHASE=33 results in an error message (ERROR=4) if the stopping
// criteria for the Krylov-Subspace iteration can not be reached. More
// information on the failure can be obtained from IPARM(20).  Note:
// The default is IPARM(4)=0 and other values are only recommended for
// advanced user.  IPARM(4) must be greater or equal to zero.
// -------------------------------------------------------------------


// IPARM(5) Use user permutation
// ------------------------------------------------------
// 0* Do not use user permutation.
// 1  Use  the user permutation provided in argument PERM
// ------------------------------------------------------


// IPARM(8) Max. numbers of iterative refinement steps.
// ---------------------------------------------------------------
// 0* Do at most k steps of iterative refinement for all matrices.
// ---------------------------------------------------------------


// IPARM(10) eps pivot (perturbation 10^-k)
// ------------------------------------------------------
// 13* Default for nonsymmetric matrices.
//  8* Default for symmetric indefinite matrices.
// ------------------------------------------------------


// IPARM(11) Use (non-) symmetric scaling vectors.
// -----------------------------------------------
//   0  Do not use
// > 1* Use (nonsymmetric matrices)
//   0* Do not use (symmetric matrices).
// -----------------------------------------------


// IPARM(13) Improved accuracy using (non-)symmetric matchings
// -----------------------------------------------------------
//  0 Do not use
// 1* Use (nonsymmetric matrices).
// 0* Do not use (symmetric matrices).
// ------------------------------------------------------


// IPARM(18) Number of nonzeros in LU.
// -------------------------------------------
//  0* Do not determine.
// -1  Will only be determined if -1 on entry.
// -------------------------------------------



// IPARM(19) Mflops for LU factorization.
// --------------------------------------------------------------------
//  0* Do not determine.
// -1  Will only be determined if -1 on entry. Increases ordering time.
// --------------------------------------------------------------------



// IPARM(21) Pivoting for symmetric indefinite matrices. Default is 1.
// -------------------------------------------------------------------
//   0  1x1 Diagonal Pivoting.
// > 1* 1x1 and 2x2 Bunch and Kaufman Pivoting.
// -------------------------------------------------------------------
  int iparm[65];
  double dparm[65];
    
// MAXFCT
// -------------------------------------------------------------------
// On entry: Maximal number of factors with identical nonzero sparsity
// structure that the user would like to keep at the same time in
// memory. It is possible to store several different factorizations
// with the same nonzero structure at the same time in the internal
// data management of the solver. In most of the applications this
// value is equal to 1.  Note: PARDISO can process several matrices
// with identical matrix sparsity pattern and is able to store the
// factors of these matrices at the same time. Matrices with different
// sparsity structure can be kept in memory with different memory
// address pointers PT.
// -------------------------------------------------------------------
  int maxfct;

    
// MNUM
// --------------------------------------------------------------------
// On entry: Actual matrix for the solution phase. With this scalar the
// user can define the matrix that he would like to factorize. The
// value must be: 1 <= MNUM <= MAXFCT. In most of the applications this
// value is equal to 1.
// --------------------------------------------------------------------
  int mnum;

    
// MTYPE
// --------------------------------------------
//  1. real and structurally symmetric
//  2. real and symmetric positive definite
// -2. real and symmetric indefinite
//  3. complex and structurally symmetric
//  4. complex and Hermitian positive definite
// -4. complex and Hermitian indefinite
//  6. complex and symmetric
// 11. real and nonsymmetric matrix
// 13. complex and nonsymmetric
// --------------------------------------------
  int mtype;

    
// MSGLVL
// ----------------------------------------------------------------
// On entry: Message level information. If MSGLVL = 0 then PARDISO
// generates no output, if MSGLVL = 1 the solver prints statistical
// information to the screen.
// ----------------------------------------------------------------
  int msglvl;

    
// PHASE
// -----------------------------------------------------------------------
// On entry: PHASE controls the execution of the solver. It is a
// two-digit integer ij (10i+j, 1 <= i <= 3, // i < j <= 3 for normal
// execution modes). The i digit indicates the starting phase of
// execution, and j indicates the ending phase. PARDISO has the
// following phases of execution:
// 1. Phase 1: Fill-reduction analysis and symbolic factorization
// 2. Phase 2: Numerical factorization  
// 3. Phase 3: Forward and Backward solve including iterative refinements
// 4. Termination and Memory Release Phase (PHASE <= 0)
//
// PHASE | Solver Execution Steps
// -----------------------------------------------------------------------    
// 11    | analysis
// 12    | analysis, numerical factorization
// 13    | analysis, numerical factorization, solve, iterative refinement
// 22    | numerical factorization
// 23    | numerical factorization, solve, iterative refinement
// 33    | solve, iterative refinement
// 0     | release L and U memory for internal matrix number MNUM
// -1    | release all internal memory for all matrices    
// -----------------------------------------------------------------------
  int phase;

    
// ERROR
// ----------
// On output: The error indicator
// ------------------------------
// Error    | Information
// ------------------------------------------------------------------------------
// 0        | no error
// -1       | input inconsistent
// -2       | not enough memory    
// -3       | reordering problem
// -4       | zero pivot, numerical factorization or iterative refinement problem
// -5       | unclassified (internal) error
// -6       | preordering failed (matrix type 11, 13 only)
// -7       | diagonal matrix problem
// -8       | 32 bit integer overflow problem
// ------------------------------------------------------------------------------
  int error;

// PERM (N)
// -------------------------------------------------------------------
// On entry: The user can supply his own fill-in reducing ordering to
// the solver. This permutation vector is only accessed if IPARM(5) =
// 1.  Note: The permutation PERM must be a vector of size N. Let A be
// the original matrix and B = PAPT be the permuted matrix. The array
// PERM is defined as follows. Row (column) of A is the PERM(i) row
// (column) of B. The numbering of the array must start by 1 and it
// must describe a permutation.
// -------------------------------------------------------------------
  int* perm;

// NRHS
// ----------------------------------------------------------------    
// On entry: NRHS is the number of right-hand sides that need to be
// solved for
// ----------------------------------------------------------------        
  int nrhs;

// N
// -----------------------------------------------------------------    
// On entry: Number of equations. This is the number of equations in
// the sparse linear systems of equations A × X = B.
// -----------------------------------------------------------------        
  int n;
    

// solver
// -----------------------------------------------------------------
// On entry: This scalar value defines the solver method the user
// would like to use:
// 
// 0: sparse direct solver
// -----------------------------------------------------------------
  int solver;




private:
  void shiftIndices_(CSRdouble& A, int value);
  void error_() const;    
  void clear_(PardisoMemoryGroup memory_to_release);



public:
  ~ParDiSO();
  ParDiSO();
  ParDiSO(int pardiso_mtype, int pardiso_msglvl, const char* filename);
  
  double memoryAllocated() const;
  void clear();
  void clearall();


  // -------------------------------
  // drivers using double arithmetic
  // -------------------------------
  void readParameters(PardisoParameters* pparams);
  void init(CSRdouble& A, int number_of_rhs);
  void factorize(CSRdouble& A);
  void solve(CSRdouble& A, double* x, double* rhs, int number_of_rhs);
  void makeSchurComplement(CSRdouble& A, CSRdouble& S);
};


#endif
