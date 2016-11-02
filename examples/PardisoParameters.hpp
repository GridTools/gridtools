#ifndef pardisoparameters_hpp
#define pardisoparameters_hpp

// include files
// =============
#include "readdata.hpp"
// =============

class PardisoParameters : public ReadData
{
private:
    virtual void printVariables_() const;
    virtual void chooseDefaultParameters_();

public:
// ########################################################################
// ########################################################################
// # Library Sparse Linear Solvers Parameters                             #
// ########################################################################
// ########################################################################



// # librarysolver may be
// # ----------------------------------------------------------------------
// # 1. PARDISO
// # 2. UMFPACK
// # 3. CHOLMOD
// # 4. ILUPACK
// #
// # DEFAULT                                                      (PARDISO)
// # ----------------------------------------------------------------------

    string librarysolver;
    int    verboselevel;
    int    maxiterations;
    double tolerance;

// ########################################################################
// # PARDISO Parameters                                                   #
// ########################################################################



// # MAXFCT           (Do not change it unless you know what you are doing)
// # ----------------------------------------------------------------------
// # On entry: Maximal number of factors with identical nonzero sparsity
// # structure that the user would like to keep at the same time in
// # memory. It is possible to store several different factorizations
// # with the same nonzero structure at the same time in the internal
// # data management of the solver. In most of the applications this
// # value is equal to 1.  Note: PARDISO can process several matrices
// # with identical matrix sparsity pattern and is able to store the
// # factors of these matrices at the same time. Matrices with different
// # sparsity structure can be kept in memory with different memory
// # address pointers PT.
// # ----------------------------------------------------------------------
    int pardiso_maxfct;


// # MNUM             (Do not change it unless you know what you are doing)
// # ----------------------------------------------------------------------
// # On entry: Actual matrix for the solution phase. With this scalar the
// # user can define the matrix that he would like to factorize. The
// # value must be: 1 <= MNUM <= MAXFCT. In most of the applications this
// # value is equal to 1.
// # ----------------------------------------------------------------------
    int pardiso_mnum;


// # MTYPE
// # ----------------------------------------------------------------------
// #  1. real and structurally symmetric
// #  2. real and symmetric positive definite
// # -2. real and symmetric indefinite
// #  3. complex and structurally symmetric
// #  4. complex and Hermitian positive definite
// # -4. complex and Hermitian indefinite
// #  6. complex and symmetric
// # 11. real and nonsymmetric matrix
// # 13. complex and nonsymmetric
// # ----------------------------------------------------------------------
    int pardiso_mtype;


// # MSGLVL
// # ----------------------------------------------------------------------
// # On entry: Message level information. If MSGLVL = 0 then PARDISO
// # generates no output, if MSGLVL = 1 the solver prints statistical
// # information to the screen.
// # ----------------------------------------------------------------------
    int pardiso_msglvl;


// # IPARM
// # ----------------------------------------------------------------------
// # On entry: IPARM is an integer array of size 64 that is used to pass
// # various parameters to PARDISO and to return some useful information
// # after the execution of the solver. PARDISOINIT fills IPARM(1),
// # IPARM(2), and IPARM(4) through IPARM(64) with default values and
// # uses them.  See section 2.3 for a detailed description.  Note: Note
// # that there is no default value for IPARM(3), which reflecsts the
// # number of processors and this value must always be supplied by the
// # user
//
// # IPARM(1) Use default options.
// # ----------------------------------------------------------------------
// # 0* Set all entries to their default values except IPARM(3)
// # ----------------------------------------------------------------------


// # IPARM(2) Use METIS reordering.
// # ----------------------------------------------------------------------
// # 0  Do not use METIS.
// # 2* Use METIS nested dissection reordering
// # ----------------------------------------------------------------------


// # IPARM(3) Number of processors.
// # ----------------------------------------------------------------------
// # p Number of OPENMP threads. This must be identical to the
// #   environment variable OMP NUM THREADS.
// # ----------------------------------------------------------------------


// # IPARM(4) Do preconditioned CGS iterations. Default is 0.
// # ----------------------------------------------------------------------
//
// # On entry: This parameter controls preconditioned CGS [12] for
// # nonsymmetric or structural symmetric matrices and
// # Conjugate-Gradients for symmetric matrices. IPARM(4) has the form
// # IPARM(4) = 10 * L + K.
//
// #################
// # K Description #
// #################
// # 0 The factorization is always computed as required by PHASE.
// # 1 CGS iteration replaces the computation of LU.  The preconditioner
// #   is LU that was computed at a previous step (the first step or last
// #   step with a failure) in a sequence of solutions needed for identical
// #   sparsity patterns.
// # 2 CG iteration for symmetric matrices replaces the computation of
// #   LU.  The preconditioner is LU that was computed at a previous step
// #   (the first step or last step with a failure) in a sequence of
// #   solutions needed for identical sparsity patterns.
//
// #################
// # L Description #
// #################
// # The value L controls the stopping criterion of the Krylov-Subspace
// # iteration: error_CGS = 10^{-L} is used in the stopping criterion
// # ||dx_i||/||dx_1|| < error_CGS with ||dx_i|| = ||(LU)^{-1}r_i|| and
// # r_i is the residuum at iteration i of the preconditioned
// # Krylov-Subspace iteration.
//
// ############
// # Strategy #
// ############
// # A maximum number of 150 iterations is fixed by expecting that the
// # iteration will converge before consuming half the factorization
// # time. Intermediate convergence rates and residuum excursions are
// # checked and can terminate the iteration process. If PHASE=23, then
// # the factorization for a given A is automatically recomputed in these
// # caseswhere the Krylov-Subspace iteration failed and the
// # corresponding direct solution is returned. Otherwise the solution
// # from the preconditioned Krylov Subspace iteration is returned. Using
// # PHASE=33 results in an error message (ERROR=4) if the stopping
// # criteria for the Krylov-Subspace iteration can not be reached. More
// # information on the failure can be obtained from IPARM(20).  Note:
// # The default is IPARM(4)=0 and other values are only recommended for
// # advanced user.  IPARM(4) must be greater or equal to zero.
// # ----------------------------------------------------------------------


// # IPARM(5) Use user permutation
// # ----------------------------------------------------------------------
// # 0* Do not use user permutation.
// # 1  Use  the user permutation provided in argument PERM
// # ----------------------------------------------------------------------


// # IPARM(8) Max. numbers of iterative refinement steps.
// # ----------------------------------------------------------------------
// # 0* Do at most k steps of iterative refinement for all matrices.
// # ----------------------------------------------------------------------


// # IPARM(10) eps pivot (perturbation 10^-k)
// # ----------------------------------------------------------------------
// # 13* Default for nonsymmetric matrices.
// #  8* Default for symmetric indefinite matrices.
// # ----------------------------------------------------------------------


// # IPARM(11) Use (non-) symmetric scaling vectors.
// # ----------------------------------------------------------------------
// #   0  Do not use
// # > 1* Use (nonsymmetric matrices)
// #   0* Do not use (symmetric matrices).
// # ----------------------------------------------------------------------


// # IPARM(13) Improved accuracy using (non-)symmetric matchings
// # ----------------------------------------------------------------------
// #  0 Do not use
// # 1* Use (nonsymmetric matrices).
// # 0* Do not use (symmetric matrices).
// # ----------------------------------------------------------------------


// # IPARM(18) Number of nonzeros in LU.
// # ----------------------------------------------------------------------
// #  0* Do not determine.
// # -1  Will only be determined if -1 on entry.
// # ----------------------------------------------------------------------


// # IPARM(19) Mflops for LU factorization.
// # ----------------------------------------------------------------------
// #  0* Do not determine.
// # -1  Will only be determined if -1 on entry. Increases ordering time.
// # ----------------------------------------------------------------------


// # IPARM(21) Pivoting for symmetric indefinite matrices. Default is 1.
// # ----------------------------------------------------------------------
// #   0  1x1 Diagonal Pivoting.
// # > 1* 1x1 and 2x2 Bunch and Kaufman Pivoting.
// # ----------------------------------------------------------------------



// # IPARM(29) Solve in 32bit mode Real*4 or float instead of double.
// # ----------------------------------------------------------------------
// # 0*   REAL*8 (double precision)
// # 1    REAL*4 (single precision)
// # ----------------------------------------------------------------------



// # IPARM(32) Use the multirecursive iterative linear solver in pardiso.
// # ----------------------------------------------------------------------
// # 0*   Use sparse direct linear solver
// # 1    Use the multi-recursive iterative solver
// # ----------------------------------------------------------------------



    int pardiso_iparm[65];


#ifdef WITH_UMFPACK
// ########################################################################
// # UMFPACK Parameters                                                   #
// ########################################################################


// # ANSI C                               default    description
// # ----------------------------------------------------------------------


// # Control[UMFPACK_PRL]                 1          printing level
// # ----------------------------------------------------------------------
// # You can control how much the umfpack_*_report_* routines print by
// # modifying the Control [UMFPACK PRL] parameter. Its default value is
// # 1. Here is a summary of how the routines use this print level
// # parameter:
// #
// # @ umfpack_*_report_status: No output if the print level is 0 or
// # less, even when an error occurs. If 1, then error messages are
// # printed, and nothing is printed if the status is UMFPACK OK. A
// # warning message is printed if the matrix is singular. If 2 or more,
// # then the status is always printed. If 4 or more, then the UMFPACK
// # Copyright is printed. If 6 or more, then the UMFPACK License is
// # printed.  See also the first page of this User Guide for the
// # Copyright and License.
// #
// # @ umfpack_*_report_control: No output if the print level is 1 or
// # less. If 2 or more, all of Control is printed.
// #
// # @ umfpack_*_report_info: No output if the print level is 1 or
// # less. If 2 or more, all of Info is printed.
// #
// # @ all other umfpack_*_report_* routines: If the print level is 2 or
// # less, then these routines return silently without checking their
// # inputs.  If 3 or more, the inputs are fully verified and a short
// # status summary is printed. If 4, then the first few entries of the
// # input arguments are printed. If 5, then all of the input arguments
// # are printed.  This print level parameter has an additional effect on
// # the MATLAB mexFunction. If zero, then no warnings of singular or
// # nearly singular matrices are printed (similar to the MATLAB commands
// # warning off MATLAB:singularMatrix and warning off
// # MATLAB:nearlySingularMatrix).
// # ----------------------------------------------------------------------
    double umfpack_prl;


// # Control[UMFPACK_DENSE_ROW]           0.2        dense row parameter
// # ----------------------------------------------------------------------
// # Rows with more than max (16, Control [UMFPACK_DENSE_ROW] * 16 * sqrt
// # (n_col)) entries are treated differently in the COLAMD pre-ordering,
// # and in the internal data structures during the subsequent numeric
// # factorization. Default: 0.2.
// # ----------------------------------------------------------------------
    double umfpack_dense_row;


// # Control[UMFPACK_DENSE_COL]           0.2        dense column parameter
// # ----------------------------------------------------------------------
// # If COLAMD is used, columns with more than max (16, Control
// # [UMFPACK_DENSE_COL] * 16 * sqrt (n_row)) entries are placed placed
// # last in the column pre-ordering. Default: 0.2.
// # ----------------------------------------------------------------------
    double umfpack_dense_col;


// # Control[UMFPACK_PIVOT_TOLERANCE]     0.1    partial pivoting tolerance
// # ----------------------------------------------------------------------
// # relative pivot tolerance for threshold partial pivoting with row
// # interchanges. In any given column, an entry is numerically
// # acceptable if its absolute value is greater than or equal to Control
// # [UMFPACK_PIVOT_TOLERANCE] times the largest absolute value in the
// # column. A value of 1.0 gives true partial pivoting. If less than or
// # equal to zero, then any nonzero entry is numerically acceptable as a
// # pivot. Default: 0.1.  Smaller values tend to lead to sparser LU
// # factors, but the solution to the linear system can become
// # inaccurate. Larger values can lead to a more accurate solution (but
// # not always), and usually an increase in the total work.  For complex
// # matrices, a cheap approximate of the absolute value is used for the
// # threshold partial pivoting test (|a_real| + |a_imag| instead of the
// # more expensive-to-compute exact absolute value sqrt (a_real^2 +
// # a_imag^2)).t
// # ----------------------------------------------------------------------
    double umfpack_pivot_tolerance;


// # Control[UMFPACK_BLOCK_SIZE]          32         BLAS block size
// # ----------------------------------------------------------------------
// # the block size to use for Level-3 BLAS in the subsequent numerical
// # factorization (umfpack_*_numeric).  A value less than 1 is treated
// # as 1. Default: 32. Modifying this parameter affects when updates are
// # applied to the working frontal matrix, and can indirectly affect
// # fill-in and operation count.  Assuming the block size is large
// # enough (8 or so), this parameter has a modest effect on performance.
// # ----------------------------------------------------------------------
    double umfpack_block_size;


// # Control[UMFPACK_STRATEGY]            0          (auto) select strategy
// # ----------------------------------------------------------------------
// # This is the most important control parameter. It determines what
// # kind of ordering and pivoting strategy that UMFPACK should
// # use. There are 4 options:
// #
// # UMFPACK_STRATEGY_AUTO: This is the default. The input matrix is
// # analyzed to determine how symmetric the nonzero pattern is, and how
// # many entries there are on the diagonal. It then selects one of the
// # following strategies. Refer to the User Guide for a description of
// # how the strategy is automatically selected.
// #
// # UMFPACK_STRATEGY_UNSYMMETRIC: Use the unsymmetric strategy. COLAMD
// # is used to order the columns of A, followed by a postorder of the
// # column elimination tree. No attempt is made to perform diagonal
// # pivoting. The column ordering is refined during factorization.  In
// # the numerical factorization, the Control
// # [UMFPACK_SYM_PIVOT_TOLERANCE] parameter is ignored. A pivot is
// # selected if its magnitude is >= Control [UMFPACK_PIVOT_TOLERANCE]
// # (default 0.1) times the largest entry in its column.
// #
// # UMFPACK_STRATEGY_SYMMETRIC: Use the symmetric strategy In this
// # method, the approximate minimum degree ordering (AMD) is applied to
// # A+A\u2019, followed by a postorder of the elimination tree of
// # A+A\u2019. UMFPACK attempts to perform diagonal pivoting during
// # numerical factorization. No refinement of the column pre-ordering is
// # performed during factorization.  In the numerical factorization, a
// # nonzero entry on the diagonal is selected as the pivot if its
// # magnitude is >= Control [UMFPACK_SYM_PIVOT_TOLERANCE] (default
// # 0.001) times the largest entry in its column. If this is not
// # acceptable, then an off-diagonal pivot is selected with magnitude >=
// # Control [UMFPACK_PIVOT_TOLERANCE] (default 0.1) times the largest
// # entry in its column.
// #
// # UMFPACK_STRATEGY_2BY2: a row permutation P2 is found that places
// # large entries on the diagonal. The matrix P2*A is then factorized
// # using the symmetric strategy, described above.  Refer to the User
// # Guide for more information.
// # ----------------------------------------------------------------------
    double umfpack_strategy;


// # Control[UMFPACK_ALLOC_INIT]          0.7     initial memory allocation
// # ----------------------------------------------------------------------
// # When umfpack_*_numeric starts, it allocates memory for the Numeric
// # object. Part of this is of fixed size (approximately n doubles +
// # 12*n integers). The remainder is of variable size, which grows to
// # hold the LU factors and the frontal matrices created during
// # factorization. A estimate of the upper bound is computed by
// # umfpack_*_*symbolic, and returned by umfpack_*_*symbolic in Info
// # [UMFPACK_VARIABLE_PEAK_ESTIMATE] (in Units).  If Control
// # [UMFPACK_ALLOC_INIT] is >= 0, umfpack_*_numeric initially allocates
// # space for the variable-sized part equal to this estimate times
// # Control [UMFPACK_ALLOC_INIT]. Typically, for matrices for which the
// # "unsymmetric" strategy applies, umfpack_*_numeric needs only about
// # half the estimated memory space, so a setting of 0.5 or 0.6 often
// # provides enough memory for umfpack_*_numeric to factorize the matrix
// # with no subsequent increases in the size of this block.  If the
// # matrix is ordered via AMD, then this non-negative parameter is
// # ignored. The initial allocation ratio computed automatically, as 1.2
// # * (nz + Info [UMFPACK_SYMMETRIC_LUNZ]) / (Info
// # [UMFPACK_LNZ_ESTIMATE] + Info [UMFPACK_UNZ_ESTIMATE] - min (n_row,
// # n_col)).  If Control [UMFPACK_ALLOC_INIT] is negative, then
// # umfpack_*_numeric allocates a space with initial size (in Units)
// # equal to (-Control [UMFPACK_ALLOC_INIT]).  Regardless of the value
// # of this parameter, a space equal to or greater than the the bare
// # minimum amount of memory needed to start the factorization is always
// # initially allocated. The bare initial memory required is returned by
// # umfpack_*_*symbolic in Info [UMFPACK_VARIABLE_INIT_ESTIMATE] (an
// # exact value, not an estimate).  If the variable-size part of the
// # Numeric object is found to be too small sometime after numerical
// # factorization has started, the memory is increased in size by a
// # factor of 1.2. If this fails, the request is reduced by a factor of
// # 0.95 until it succeeds, or until it determines that no increase in
// # size is possible. Garbage collection then occurs.  The strategy of
// # attempting to "malloc" a working space, and re-trying with a smaller
// # space, may not work when UMFPACK is used as a mexFunction MATLAB,
// # since mxMalloc aborts the mexFunction if it fails. This issue does
// # not affect the use of UMFPACK as a part of the built-in x=A\b in
// # MATLAB 6.5 and later.  If you are using the umfpack mexFunction,
// # decrease the magnitude of Control [UMFPACK_ALLOC_INIT] if you run
// # out of memory in MATLAB.  Default initial allocation size:
// # 0.7. Thus, with the default control settings and the "unsymmetric"
// # strategy, the upper-bound is reached after two reallocations (0.7 *
// # 1.2 * 1.2 = 1.008).  Changing this parameter has little effect on
// # fill-in or operation count. It has a small impact on run-time (the
// # extra time required to do the garbage collection and memory
// # reallocation).
// # ----------------------------------------------------------------------
    double umfpack_alloc_init;


// # Control[UMFPACK_IRSTEP]              2      max iter. refinement steps
// # ----------------------------------------------------------------------
// # The maximum number of iterative refinement steps to attempt. A value
// # less than zero is treated as zero. If less than 1, or if Ax=b,
// # A'x=b, or A.'x=b is not being solved, or if A is singular, then the
// # Ap, Ai, Ax, and Az arguments are not accessed.
// # ----------------------------------------------------------------------
    double umfpack_irstep;


// # Control[UMFPACK_2BY2_TOLERANCE]      0.01       defines "large" entries
// # ----------------------------------------------------------------------
// # a diagonal entry S (k,k) is considered "small" if it is < tol * max
// # (abs (S (:,k))), where S a submatrix of the scaled input matrix,
// # with pivots of zero Markowitz cost removed.
// # ----------------------------------------------------------------------
    double umfpack_twobytwo_tolerance;


// # Control[UMFPACK_FIXQ]                0          (auto) fix or modify Q
// # ----------------------------------------------------------------------
// # If > 0, then the pre-ordering Q is not modified during numeric
// # factorization. If < 0, then Q may be modified. If zero, then this is
// # controlled automatically (the unsymmetric strategy modifies Q, the
// # others do not). Default: 0.
// # ----------------------------------------------------------------------
    double umfpack_fixq;


// # Control[UMFPACK_AMD_DENSE]           10 AMD dense row/column parameter
// # ----------------------------------------------------------------------
// # rows/columns in A+A' with more than max (16, Control
// # [UMFPACK_AMD_DENSE] * sqrt (n)) entries (where n = n_row = n_col)
// # are ignored in the AMD pre-ordering.  Default: 10
// # ----------------------------------------------------------------------
    double umfpack_amd_dense;


// # Control[UMFPACK_SYM_PIVOT_TOLERANCE] 0.001      for diagonal entries
// # ----------------------------------------------------------------------
// # If diagonal pivoting is attempted (the symmetric or symmetric-2by2
// # strategies are used) then this parameter is used to control when the
// # diagonal entry is selected in a given pivot column. The absoluter
// # value of the entry must be >= Control [UMFPACK_SYM_PIVOT_TOLERANCE]
// # times the largest absolute value in the column. A value of zero will
// # ensure that no off-diagonal pivoting is performed, except that zero
// # diagonal entries are not selected if there are any off-diagonal
// # nonzero entries.  If an off-diagonal pivot is selected, an attempt
// # is made to restore symmetry later on. Suppose A (i,j) is selected,
// # where i != j.  If column i has not yet been selected as a pivot
// # column, then the entry A (j,i) is redefined as a "diagonal" entry,
// # except that the tighter tolerance (Control
// # [UMFPACK_PIVOT_TOLERANCE]) is applied. This strategy has an effect
// # similar to 2-by-2 pivoting for symmetric indefinite matrices. If a
// # 2-by-2 block pivot with nonzero structure
// #
// #    i j
// # i: 0 x
// # j: x 0
// #
// # is selected in a symmetric indefinite factorization method, the
// # 2-by-2 # block is inverted and a rank-2 update is applied. In
// # UMFPACK, this # 2-by-2 block would be reordered as
// #
// #    j i
// # i: x 0
// # j: 0 x
// #
// # In both cases, the symmetry of the Schur complement is preserved.
// # ----------------------------------------------------------------------
    double umfpack_sym_pivot_tolerance;


// # Control[UMFPACK_SCALE]     1     (sum) row scaling (none, sum, or max)
// # ----------------------------------------------------------------------
// # Note that the user's input matrix is never modified, only an
// # internal copy is scaled.  There are three valid settings for this
// # parameter. If any other value is provided, the default is used.
// #
// # UMFPACK_SCALE_NONE: no scaling is performed.
// #
// # UMFPACK_SCALE_SUM: each row of the input matrix A is divided by the
// # sum of the absolute values of the entries in that row.  The scaled
// # matrix has an infinity norm of 1.
// #
// # UMFPACK_SCALE_MAX: each row of the input matrix A is divided by the
// # maximum the absolute values of the entries in that row.  In the
// # scaled matrix the largest entry in each row has a magnitude exactly
// # equal to 1.  Note that for complex matrices, a cheap approximate
// # absolute value is used, |a_real| + |a_imag|, instead of the exact
// # absolute value sqrt ((a_real)^2 + (a_imag)^2).  Scaling is very
// # important for the "symmetric" strategy when diagonal pivoting is
// # attempted. It also improves the performance of the "unsymmetric"
// # strategy.
// #
// # Default: UMFPACK_SCALE_SUM.
// # ----------------------------------------------------------------------
    double umfpack_scale;


// # Control[UMFPACK_FRONT_ALLOC_INIT]  0.5 frontal matrix allocation ratio
// # ----------------------------------------------------------------------
// # When UMFPACK starts the factorization of each "chain" of frontala
// # matrices, it allocates a working array to hold the frontal matrices
// # as they are factorized. The symbolic factorization computes the size
// # of the largest possible frontal matrix that could occur during the
// # factorization of each chain.  If Control [UMFPACK_FRONT_ALLOC_INIT]
// # is >= 0, the following strategy is used. If the AMD ordering was
// # used, this non-negative parameter is ignored. A front of size
// # (d+2)*(d+2) is allocated, where d = Info
// # [UMFPACK_SYMMETRIC_DMAX]. Otherwise, a front of size Control
// # [UMFPACK_FRONT_ALLOC_INIT] times the largest front possible for this
// # chain is allocated.  If Control [UMFPACK_FRONT_ALLOC_INIT] is
// # negative, then a front of size (-Control [UMFPACK_FRONT_ALLOC_INIT])
// # is allocated (where the size is in terms of the number of numerical
// # entries). This is done regardless of the ordering method or ordering
// # strategy used.
// #
// # Default: 0.5.
// # ----------------------------------------------------------------------
    double umfpack_front_alloc_init;


// # Control[UMFPACK_DROPTOL]             0                  drop tolerance
// # ----------------------------------------------------------------------
// # Entries in L and U with absolute value less than or equal to the
// # drop tolerance are removed from the data structures (unless leaving
// # them there reduces memory usage by reducing the space required for
// # the nonzero pattern of L and U).  Default: 0.0.
// # ----------------------------------------------------------------------
    double umfpack_droptol;


// # Control[UMFPACK_AGGRESSIVE]          1     (yes) aggressive absorption
// #                                                 in  AMD and COLAMD
// # ----------------------------------------------------------------------
// # If nonzero, aggressive absorption is used in COLAMD and
// # AMD.
// #
// # Default: 1.
// # ----------------------------------------------------------------------
    double umfpack_aggressive;

#endif


public:
    ~PardisoParameters();
    PardisoParameters();
    PardisoParameters(string datafile);
    PardisoParameters(fstream& fin);
    virtual void init(string datafile);
    virtual void init(fstream& fin);
};

#endif

