// include files
// =============
#include "PardisoParameters.hpp"
#include <iostream>
using std::cout;
// =============

// =================================
// ========== Destructor ===========
// =================================

PardisoParameters::~PardisoParameters()
{
}

// =================================
// ========== Contructors ==========
// =================================

PardisoParameters::PardisoParameters()
    : ReadData()
{
    chooseDefaultParameters_();
}

PardisoParameters::PardisoParameters(string datafile)
    : ReadData()
{
    chooseDefaultParameters_();

    if (datafile != "DEFAULT")
    {
        init(datafile);
        if (printparameters_)
        {
            printVariables_();
        }
    }
    else
    {
        printVariables_();
    }
}

PardisoParameters::PardisoParameters(fstream& fin)
{
    init(fin);
}
// =================================
// ======== Virtual Methods ========
// =================================

void
PardisoParameters::printVariables_() const
{
    cout << "########################################################################\n";
    cout << "########################################################################\n";
    cout << "# Library Sparse Linear Solvers Parameters                             #\n";
    cout << "########################################################################\n";
    cout << "########################################################################\n";



    cout << "\n\n\n\n";

    cout << "# librarysolver may be\n";
    cout << "# ----------------------------------------------------------------------\n";
    cout << "# 1. PARDISO\n";
    cout << "# 2. UMFPACK\n";
    cout << "# 3. CHOLMOD\n";
    cout << "# 4. ILUPACK\n";
    cout << "#\n";
    cout << "# DEFAULT                                                      (PARDISO)\n";
    cout << "# ----------------------------------------------------------------------\n";

    printVariable_("librarysolver", librarysolver);
    printVariable_("verboselevel", verboselevel);
    printVariable_("maxiterations", maxiterations);
    printVariable_("tolerance", tolerance);




    cout << "########################################################################\n";
    cout << "# PARDISO Parameters                                                   #\n";
    cout << "########################################################################\n";


    cout << "\n";
    cout << "\n";
    cout << "\n";


    cout << "# MAXFCT           (Do not change it unless you know what you are doing)\n";
    cout << "# ----------------------------------------------------------------------\n";
    cout << "# On entry: Maximal number of factors with identical nonzero sparsity\n";
    cout << "# structure that the user would like to keep at the same time in\n";
    cout << "# memory. It is possible to store several different factorizations\n";
    cout << "# with the same nonzero structure at the same time in the internal\n";
    cout << "# data management of the solver. In most of the applications this\n";
    cout << "# value is equal to 1.  Note: PARDISO can process several matrices\n";
    cout << "# with identical matrix sparsity pattern and is able to store the\n";
    cout << "# factors of these matrices at the same time. Matrices with different\n";
    cout << "# sparsity structure can be kept in memory with different memory\n";
    cout << "# address pointers PT.\n";
    cout << "# ----------------------------------------------------------------------\n";

    printVariable_("pardiso_maxfct", pardiso_maxfct);


    cout << "# MNUM             (Do not change it unless you know what you are doing)\n";
    cout << "# ----------------------------------------------------------------------\n";
    cout << "# On entry: Actual matrix for the solution phase. With this scalar the\n";
    cout << "# user can define the matrix that he would like to factorize. The\n";
    cout << "# value must be: 1 <= MNUM <= MAXFCT. In most of the applications this\n";
    cout << "# value is equal to 1.\n";
    cout << "# ----------------------------------------------------------------------\n";

    printVariable_("pardiso_mnum", pardiso_mnum);


    cout << "# MTYPE\n";
    cout << "# ----------------------------------------------------------------------\n";
    cout << "#  1. real and structurally symmetric\n";
    cout << "#  2. real and symmetric positive definite\n";
    cout << "# -2. real and symmetric indefinite\n";
    cout << "#  3. complex and structurally symmetric\n";
    cout << "#  4. complex and Hermitian positive definite\n";
    cout << "# -4. complex and Hermitian indefinite\n";
    cout << "#  6. complex and symmetric\n";
    cout << "# 11. real and nonsymmetric matrix\n";
    cout << "# 13. complex and nonsymmetric\n";
    cout << "# ----------------------------------------------------------------------\n";


    printVariable_("pardiso_mtype", pardiso_mtype);


    cout << "# MSGLVL\n";
    cout << "# ----------------------------------------------------------------------\n";
    cout << "# On entry: Message level information. If MSGLVL = 0 then PARDISO\n";
    cout << "# generates no output, if MSGLVL = 1 the solver prints statistical\n";
    cout << "# information to the screen.\n";
    cout << "# ----------------------------------------------------------------------\n";


    printVariable_("pardiso_msglvl", pardiso_msglvl);


    cout << "# IPARM\n";
    cout << "# ----------------------------------------------------------------------\n";
    cout << "# On entry: IPARM is an integer array of size 64 that is used to pass\n";
    cout << "# various parameters to PARDISO and to return some useful information\n";
    cout << "# after the execution of the solver. PARDISOINIT fills IPARM(1),\n";
    cout << "# IPARM(2), and IPARM(4) through IPARM(64) with default values and\n";
    cout << "# uses them.  See section 2.3 for a detailed description.  Note: Note\n";
    cout << "# that there is no default value for IPARM(3), which reflecsts the\n";
    cout << "# number of processors and this value must always be supplied by the\n";
    cout << "# user\n";
    cout << "\n";
    cout << "# IPARM(1) Use default options.\n";
    cout << "# ----------------------------------------------------------------------\n";
    cout << "# 0* Set all entries to their default values except IPARM(3)\n";
    cout << "# ----------------------------------------------------------------------\n";


    printVariable_("pardiso_iparm[1]", pardiso_iparm[1]);


    cout << "# IPARM(2) Use METIS reordering.\n";
    cout << "# ----------------------------------------------------------------------\n";
    cout << "# 0  Do not use METIS.\n";
    cout << "# 2* Use METIS nested dissection reordering\n";
    cout << "# ----------------------------------------------------------------------\n";


    printVariable_("pardiso_iparm[2]", pardiso_iparm[2]);


    cout << "# IPARM(3) Number of processors.\n";
    cout << "# ----------------------------------------------------------------------\n";
    cout << "# p Number of OPENMP threads. This must be identical to the\n";
    cout << "#   environment variable OMP NUM THREADS.\n";
    cout << "# ----------------------------------------------------------------------\n";


    printVariable_("pardiso_iparm[3]", pardiso_iparm[3]);


    cout << "# IPARM(4) Do preconditioned CGS iterations. Default is 0.\n";
    cout << "# ----------------------------------------------------------------------\n";
    cout << "\n";
    cout << "# On entry: This parameter controls preconditioned CGS [12] for\n";
    cout << "# nonsymmetric or structural symmetric matrices and\n";
    cout << "# Conjugate-Gradients for symmetric matrices. IPARM(4) has the form\n";
    cout << "# IPARM(4) = 10 * L + K.\n";
    cout << "\n";
    cout << "#################\n";
    cout << "# K Description #\n";
    cout << "#################\n";
    cout << "# 0 The factorization is always computed as required by PHASE.\n";
    cout << "# 1 CGS iteration replaces the computation of LU.  The preconditioner\n";
    cout << "#   is LU that was computed at a previous step (the first step or last\n";
    cout << "#   step with a failure) in a sequence of solutions needed for identical\n";
    cout << "#   sparsity patterns.\n";
    cout << "# 2 CG iteration for symmetric matrices replaces the computation of\n";
    cout << "#   LU.  The preconditioner is LU that was computed at a previous step\n";
    cout << "#   (the first step or last step with a failure) in a sequence of\n";
    cout << "#   solutions needed for identical sparsity patterns.\n";
    cout << "\n";
    cout << "#################\n";
    cout << "# L Description #\n";
    cout << "#################\n";
    cout << "# The value L controls the stopping criterion of the Krylov-Subspace\n";
    cout << "# iteration: error_CGS = 10^{-L} is used in the stopping criterion\n";
    cout << "# ||dx_i||/||dx_1|| < error_CGS with ||dx_i|| = ||(LU)^{-1}r_i|| and\n";
    cout << "# r_i is the residuum at iteration i of the preconditioned\n";
    cout << "# Krylov-Subspace iteration.\n";
    cout << "\n";
    cout << "############\n";
    cout << "# Strategy #\n";
    cout << "############\n";
    cout << "# A maximum number of 150 iterations is fixed by expecting that the\n";
    cout << "# iteration will converge before consuming half the factorization\n";
    cout << "# time. Intermediate convergence rates and residuum excursions are\n";
    cout << "# checked and can terminate the iteration process. If PHASE=23, then\n";
    cout << "# the factorization for a given A is automatically recomputed in these\n";
    cout << "# caseswhere the Krylov-Subspace iteration failed and the\n";
    cout << "# corresponding direct solution is returned. Otherwise the solution\n";
    cout << "# from the preconditioned Krylov Subspace iteration is returned. Using\n";
    cout << "# PHASE=33 results in an error message (ERROR=4) if the stopping\n";
    cout << "# criteria for the Krylov-Subspace iteration can not be reached. More\n";
    cout << "# information on the failure can be obtained from IPARM(20).  Note:\n";
    cout << "# The default is IPARM(4)=0 and other values are only recommended for\n";
    cout << "# advanced user.  IPARM(4) must be greater or equal to zero.\n";
    cout << "# ----------------------------------------------------------------------\n";


    printVariable_("pardiso_iparm[4]", pardiso_iparm[4]);


    cout << "# IPARM(5) Use user permutation\n";
    cout << "# ----------------------------------------------------------------------\n";
    cout << "# 0* Do not use user permutation.\n";
    cout << "# 1  Use  the user permutation provided in argument PERM\n";
    cout << "# ----------------------------------------------------------------------\n";


    printVariable_("pardiso_iparm[5]", pardiso_iparm[5]);



    cout << "IPARM (6) — Write solution on X. Inpu\n";
    cout << "# ----------------------------------------------------------------------\n";
    cout << "On entry: If IPARM(6) is 0, which is the default, then the array X contains the\n";
    cout << "solution and the value of B is not changed. If IPARM(6) is 1 then the solver\n";
    cout << "will store the solution on the right-hand side B.\n";
    cout << "Note: The array X is always changed. The default value of IPARM(6) is 0.\n";
    cout << "# ----------------------------------------------------------------------\n";

    printVariable_("pardiso_iparm[6]", pardiso_iparm[6]);


    cout << "# IPARM(8) Max. numbers of iterative refinement steps.\n";
    cout << "# ----------------------------------------------------------------------\n";
    cout << "# 0* Do at most k steps of iterative refinement for all matrices.\n";
    cout << "# ----------------------------------------------------------------------\n";


    printVariable_("pardiso_iparm[8]", pardiso_iparm[8]);


    cout << "# IPARM(10) eps pivot (perturbation 10^-k)\n";
    cout << "# ----------------------------------------------------------------------\n";
    cout << "# 13* Default for nonsymmetric matrices.\n";
    cout << "#  8* Default for symmetric indefinite matrices.\n";
    cout << "# ----------------------------------------------------------------------\n";


    printVariable_("pardiso_iparm[10]", pardiso_iparm[10]);


    cout << "# IPARM(11) Use (non-) symmetric scaling vectors.\n";
    cout << "# ----------------------------------------------------------------------\n";
    cout << "#   0  Do not use\n";
    cout << "# > 1* Use (nonsymmetric matrices)\n";
    cout << "#   0* Do not use (symmetric matrices).\n";
    cout << "# ----------------------------------------------------------------------\n";


    printVariable_("pardiso_iparm[11]", pardiso_iparm[11]);

    cout << "# IPARM(12) solve a system A^T x = b using the factorization of A\n";
    cout << "# ----------------------------------------------------------------------\n";
    cout << "# 0* Solve Ax = b\n";
    cout << "# 1  Solve A^T x = b\n";
    cout << "# ----------------------------------------------------------------------\n";

    printVariable_("pardiso_iparm[12]", pardiso_iparm[12]);

    cout << "# IPARM(13) Improved accuracy using (non-)symmetric matchings\n";
    cout << "# ----------------------------------------------------------------------\n";
    cout << "#  0 Do not use\n";
    cout << "# 1* Use (nonsymmetric matrices).\n";
    cout << "# 0* Do not use (symmetric matrices).\n";
    cout << "# ----------------------------------------------------------------------\n";


    printVariable_("pardiso_iparm[13]", pardiso_iparm[13]);


    cout << "# IPARM(18) Number of nonzeros in LU.\n";
    cout << "# ----------------------------------------------------------------------\n";
    cout << "#  0* Do not determine.\n";
    cout << "# -1  Will only be determined if -1 on entry.\n";
    cout << "# ----------------------------------------------------------------------\n";


    printVariable_("pardiso_iparm[18]", pardiso_iparm[18]);


    cout << "# IPARM(19) Mflops for LU factorization.\n";
    cout << "# ----------------------------------------------------------------------\n";
    cout << "#  0* Do not determine.\n";
    cout << "# -1  Will only be determined if -1 on entry. Increases ordering time.\n";
    cout << "# ----------------------------------------------------------------------\n";


    printVariable_("pardiso_iparm[19]", pardiso_iparm[19]);


    cout << "# IPARM(21) Pivoting for symmetric indefinite matrices. Default is 1.\n";
    cout << "# ----------------------------------------------------------------------\n";
    cout << "#   0  1x1 Diagonal Pivoting.\n";
    cout << "# > 1* 1x1 and 2x2 Bunch and Kaufman Pivoting.\n";
    cout << "# ----------------------------------------------------------------------\n";

    printVariable_("pardiso_iparm[21]", pardiso_iparm[21]);


    cout << "# IPARM(28) Parallel Reordering for METIS.\n";
    cout << "# ----------------------------------------------------------------------\n";
    cout << "#   0* execute METIS sequentially.\n";
    cout << "#   1* execute METIS in parallel.\n";
    cout << "# ----------------------------------------------------------------------\n";


    printVariable_("pardiso_iparm[28]", pardiso_iparm[28]);


    cout << "# IPARM(29) Solve in 32bit mode Real*4 or float instead of double.\n";
    cout << "# ----------------------------------------------------------------------\n";
    cout << "# 0*   REAL*8 (double precision)\n";
    cout << "# 1    REAL*4 (single precision)\n";
    cout << "# ----------------------------------------------------------------------\n";

    printVariable_("pardiso_iparm[29]", pardiso_iparm[29]);

    cout << "# IPARM(32) Use the multirecursive iterative solver in pardiso.\n";
    cout << "# ----------------------------------------------------------------------\n";
    cout << "# 0*   Use the sparse direct solver in pardiso (standard choice)\n";
    cout << "# 1    Use the multi-recursive iterative solver in pardiso\n";
    cout << "# ----------------------------------------------------------------------\n";

    printVariable_("pardiso_iparm[32]", pardiso_iparm[32]);




#ifdef WITH_UMFPACK

    cout << "########################################################################\n";
    cout << "# UMFPACK Parameters                                                   #\n";
    cout << "########################################################################\n";


    cout << "\n";
    cout << "\n";
    cout << "\n";


    cout << "# ANSI C                               default    description\n";
    cout << "# ----------------------------------------------------------------------\n";
    cout << "\n";
    cout << "\n";
    cout << "# Control[UMFPACK_PRL]                 1          printing level\n";
    cout << "# ----------------------------------------------------------------------\n";
    cout << "# You can control how much the umfpack_*_report_* routines print by\n";
    cout << "# modifying the Control [UMFPACK PRL] parameter. Its default value is\n";
    cout << "# 1. Here is a summary of how the routines use this print level\n";
    cout << "# parameter:\n";
    cout << "#\n";
    cout << "# @ umfpack_*_report_status: No output if the print level is 0 or\n";
    cout << "# less, even when an error occurs. If 1, then error messages are\n";
    cout << "# printed, and nothing is printed if the status is UMFPACK OK. A\n";
    cout << "# warning message is printed if the matrix is singular. If 2 or more,\n";
    cout << "# then the status is always printed. If 4 or more, then the UMFPACK\n";
    cout << "# Copyright is printed. If 6 or more, then the UMFPACK License is\n";
    cout << "# printed.  See also the first page of this User Guide for the\n";
    cout << "# Copyright and License.\n";
    cout << "#\n";
    cout << "# @ umfpack_*_report_control: No output if the print level is 1 or\n";
    cout << "# less. If 2 or more, all of Control is printed.\n";
    cout << "#\n";
    cout << "# @ umfpack_*_report_info: No output if the print level is 1 or\n";
    cout << "# less. If 2 or more, all of Info is printed.\n";
    cout << "#\n";
    cout << "# @ all other umfpack_*_report_* routines: If the print level is 2 or\n";
    cout << "# less, then these routines return silently without checking their\n";
    cout << "# inputs.  If 3 or more, the inputs are fully verified and a short\n";
    cout << "# status summary is printed. If 4, then the first few entries of the\n";
    cout << "# input arguments are printed. If 5, then all of the input arguments\n";
    cout << "# are printed.  This print level parameter has an additional effect on\n";
    cout << "# the MATLAB mexFunction. If zero, then no warnings of singular or\n";
    cout << "# nearly singular matrices are printed (similar to the MATLAB commands\n";
    cout << "# warning off MATLAB:singularMatrix and warning off\n";
    cout << "# MATLAB:nearlySingularMatrix).\n";
    cout << "# ----------------------------------------------------------------------\n";


    printVariable_("umfpack_prl", umfpack_prl);


    cout << "# Control[UMFPACK_DENSE_ROW]           0.2        dense row parameter\n";
    cout << "# ----------------------------------------------------------------------\n";
    cout << "# Rows with more than max (16, Control [UMFPACK_DENSE_ROW] * 16 * sqrt\n";
    cout << "# (n_col)) entries are treated differently in the COLAMD pre-ordering,\n";
    cout << "# and in the internal data structures during the subsequent numeric\n";
    cout << "# factorization. Default: 0.2.\n";
    cout << "# ----------------------------------------------------------------------\n";


    printVariable_("umfpack_dense_row", umfpack_dense_row);


    cout << "# Control[UMFPACK_DENSE_COL]           0.2        dense column parameter\n";
    cout << "# ----------------------------------------------------------------------\n";
    cout << "# If COLAMD is used, columns with more than max (16, Control\n";
    cout << "# [UMFPACK_DENSE_COL] * 16 * sqrt (n_row)) entries are placed placed\n";
    cout << "# last in the column pre-ordering. Default: 0.2.\n";
    cout << "# ----------------------------------------------------------------------\n";


    printVariable_("umfpack_dense_col", umfpack_dense_col);


    cout << "# Control[UMFPACK_PIVOT_TOLERANCE]     0.1    partial pivoting tolerance\n";
    cout << "# ----------------------------------------------------------------------\n";
    cout << "# relative pivot tolerance for threshold partial pivoting with row\n";
    cout << "# interchanges. In any given column, an entry is numerically\n";
    cout << "# acceptable if its absolute value is greater than or equal to Control\n";
    cout << "# [UMFPACK_PIVOT_TOLERANCE] times the largest absolute value in the\n";
    cout << "# column. A value of 1.0 gives true partial pivoting. If less than or\n";
    cout << "# equal to zero, then any nonzero entry is numerically acceptable as a\n";
    cout << "# pivot. Default: 0.1.  Smaller values tend to lead to sparser LU\n";
    cout << "# factors, but the solution to the linear system can become\n";
    cout << "# inaccurate. Larger values can lead to a more accurate solution (but\n";
    cout << "# not always), and usually an increase in the total work.  For complex\n";
    cout << "# matrices, a cheap approximate of the absolute value is used for the\n";
    cout << "# threshold partial pivoting test (|a_real| + |a_imag| instead of the\n";
    cout << "# more expensive-to-compute exact absolute value sqrt (a_real^2 +\n";
    cout << "# a_imag^2)).t\n";
    cout << "# ----------------------------------------------------------------------\n";


    printVariable_("umfpack_pivot_tolerance", umfpack_pivot_tolerance);


    cout << "# Control[UMFPACK_BLOCK_SIZE]          32         BLAS block size\n";
    cout << "# ----------------------------------------------------------------------\n";
    cout << "# the block size to use for Level-3 BLAS in the subsequent numerical\n";
    cout << "# factorization (umfpack_*_numeric).  A value less than 1 is treated\n";
    cout << "# as 1. Default: 32. Modifying this parameter affects when updates are\n";
    cout << "# applied to the working frontal matrix, and can indirectly affect\n";
    cout << "# fill-in and operation count.  Assuming the block size is large\n";
    cout << "# enough (8 or so), this parameter has a modest effect on performance.\n";
    cout << "# ----------------------------------------------------------------------\n";


    printVariable_("umfpack_block_size", umfpack_block_size);


    cout << "# Control[UMFPACK_STRATEGY]            0          (auto) select strategy\n";
    cout << "# ----------------------------------------------------------------------\n";
    cout << "# This is the most important control parameter. It determines what\n";
    cout << "# kind of ordering and pivoting strategy that UMFPACK should\n";
    cout << "# use. There are 4 options:\n";
    cout << "#\n";
    cout << "# UMFPACK_STRATEGY_AUTO: This is the default. The input matrix is\n";
    cout << "# analyzed to determine how symmetric the nonzero pattern is, and how\n";
    cout << "# many entries there are on the diagonal. It then selects one of the\n";
    cout << "# following strategies. Refer to the User Guide for a description of\n";
    cout << "# how the strategy is automatically selected.\n";
    cout << "#\n";
    cout << "# UMFPACK_STRATEGY_UNSYMMETRIC: Use the unsymmetric strategy. COLAMD\n";
    cout << "# is used to order the columns of A, followed by a postorder of the\n";
    cout << "# column elimination tree. No attempt is made to perform diagonal\n";
    cout << "# pivoting. The column ordering is refined during factorization.  In\n";
    cout << "# the numerical factorization, the Control\n";
    cout << "# [UMFPACK_SYM_PIVOT_TOLERANCE] parameter is ignored. A pivot is\n";
    cout << "# selected if its magnitude is >= Control [UMFPACK_PIVOT_TOLERANCE]\n";
    cout << "# (default 0.1) times the largest entry in its column.\n";
    cout << "#\n";
    cout << "# UMFPACK_STRATEGY_SYMMETRIC: Use the symmetric strategy In this\n";
    cout << "# method, the approximate minimum degree ordering (AMD) is applied to\n";
    cout << "# A+A\u2019, followed by a postorder of the elimination tree of\n";
    cout << "# A+A\u2019. UMFPACK attempts to perform diagonal pivoting during\n";
    cout << "# numerical factorization. No refinement of the column pre-ordering is\n";
    cout << "# performed during factorization.  In the numerical factorization, a\n";
    cout << "# nonzero entry on the diagonal is selected as the pivot if its\n";
    cout << "# magnitude is >= Control [UMFPACK_SYM_PIVOT_TOLERANCE] (default\n";
    cout << "# 0.001) times the largest entry in its column. If this is not\n";
    cout << "# acceptable, then an off-diagonal pivot is selected with magnitude >=\n";
    cout << "# Control [UMFPACK_PIVOT_TOLERANCE] (default 0.1) times the largest\n";
    cout << "# entry in its column.\n";
    cout << "#\n";
    cout << "# UMFPACK_STRATEGY_2BY2: a row permutation P2 is found that places\n";
    cout << "# large entries on the diagonal. The matrix P2*A is then factorized\n";
    cout << "# using the symmetric strategy, described above.  Refer to the User\n";
    cout << "# Guide for more information.\n";
    cout << "# ----------------------------------------------------------------------\n";


    printVariable_("umfpack_strategy", umfpack_strategy);


    cout << "# Control[UMFPACK_ALLOC_INIT]          0.7     initial memory allocation\n";
    cout << "# ----------------------------------------------------------------------\n";
    cout << "# When umfpack_*_numeric starts, it allocates memory for the Numeric\n";
    cout << "# object. Part of this is of fixed size (approximately n doubles +\n";
    cout << "# 12*n integers). The remainder is of variable size, which grows to\n";
    cout << "# hold the LU factors and the frontal matrices created during\n";
    cout << "# factorization. A estimate of the upper bound is computed by\n";
    cout << "# umfpack_*_*symbolic, and returned by umfpack_*_*symbolic in Info\n";
    cout << "# [UMFPACK_VARIABLE_PEAK_ESTIMATE] (in Units).  If Control\n";
    cout << "# [UMFPACK_ALLOC_INIT] is >= 0, umfpack_*_numeric initially allocates\n";
    cout << "# space for the variable-sized part equal to this estimate times\n";
    cout << "# Control [UMFPACK_ALLOC_INIT]. Typically, for matrices for which the\n";
    cout << "# \"unsymmetric\" strategy applies, umfpack_*_numeric needs only about\n";
    cout << "# half the estimated memory space, so a setting of 0.5 or 0.6 often\n";
    cout << "# provides enough memory for umfpack_*_numeric to factorize the matrix\n";
    cout << "# with no subsequent increases in the size of this block.  If the\n";
    cout << "# matrix is ordered via AMD, then this non-negative parameter is\n";
    cout << "# ignored. The initial allocation ratio computed automatically, as 1.2\n";
    cout << "# * (nz + Info [UMFPACK_SYMMETRIC_LUNZ]) / (Info\n";
    cout << "# [UMFPACK_LNZ_ESTIMATE] + Info [UMFPACK_UNZ_ESTIMATE] - min (n_row,\n";
    cout << "# n_col)).  If Control [UMFPACK_ALLOC_INIT] is negative, then\n";
    cout << "# umfpack_*_numeric allocates a space with initial size (in Units)\n";
    cout << "# equal to (-Control [UMFPACK_ALLOC_INIT]).  Regardless of the value\n";
    cout << "# of this parameter, a space equal to or greater than the the bare\n";
    cout << "# minimum amount of memory needed to start the factorization is always\n";
    cout << "# initially allocated. The bare initial memory required is returned by\n";
    cout << "# umfpack_*_*symbolic in Info [UMFPACK_VARIABLE_INIT_ESTIMATE] (an\n";
    cout << "# exact value, not an estimate).  If the variable-size part of the\n";
    cout << "# Numeric object is found to be too small sometime after numerical\n";
    cout << "# factorization has started, the memory is increased in size by a\n";
    cout << "# factor of 1.2. If this fails, the request is reduced by a factor of\n";
    cout << "# 0.95 until it succeeds, or until it determines that no increase in\n";
    cout << "# size is possible. Garbage collection then occurs.  The strategy of\n";
    cout << "# attempting to \"malloc\" a working space, and re-trying with a smaller\n";
    cout << "# space, may not work when UMFPACK is used as a mexFunction MATLAB,\n";
    cout << "# since mxMalloc aborts the mexFunction if it fails. This issue does\n";
    cout << "# not affect the use of UMFPACK as a part of the built-in x=A\b in\n";
    cout << "# MATLAB 6.5 and later.  If you are using the umfpack mexFunction,\n";
    cout << "# decrease the magnitude of Control [UMFPACK_ALLOC_INIT] if you run\n";
    cout << "# out of memory in MATLAB.  Default initial allocation size:\n";
    cout << "# 0.7. Thus, with the default control settings and the \"unsymmetric\"\n";
    cout << "# strategy, the upper-bound is reached after two reallocations (0.7 *\n";
    cout << "# 1.2 * 1.2 = 1.008).  Changing this parameter has little effect on\n";
    cout << "# fill-in or operation count. It has a small impact on run-time (the\n";
    cout << "# extra time required to do the garbage collection and memory\n";
    cout << "# reallocation).\n";
    cout << "# ----------------------------------------------------------------------\n";


    printVariable_("umfpack_alloc_init", umfpack_alloc_init);


    cout << "# Control[UMFPACK_IRSTEP]              2      max iter. refinement steps\n";
    cout << "# ----------------------------------------------------------------------\n";
    cout << "# The maximum number of iterative refinement steps to attempt. A value\n";
    cout << "# less than zero is treated as zero. If less than 1, or if Ax=b,\n";
    cout << "# A'x=b, or A.'x=b is not being solved, or if A is singular, then the\n";
    cout << "# Ap, Ai, Ax, and Az arguments are not accessed. \n";
    cout << "# ----------------------------------------------------------------------\n";


    printVariable_("umfpack_irstep", umfpack_irstep);


    cout << "# Control[UMFPACK_2BY2_TOLERANCE]      0.01       defines \"large\" entries\n";
    cout << "# ----------------------------------------------------------------------\n";
    cout << "# a diagonal entry S (k,k) is considered \"small\" if it is < tol * max\n";
    cout << "# (abs (S (:,k))), where S a submatrix of the scaled input matrix,\n";
    cout << "# with pivots of zero Markowitz cost removed.\n";
    cout << "# ----------------------------------------------------------------------\n";


    printVariable_("umfpack_twobytwo_tolerance", umfpack_twobytwo_tolerance);


    cout << "# Control[UMFPACK_FIXQ]                0          (auto) fix or modify Q\n";
    cout << "# ----------------------------------------------------------------------\n";
    cout << "# If > 0, then the pre-ordering Q is not modified during numeric\n";
    cout << "# factorization. If < 0, then Q may be modified. If zero, then this is\n";
    cout << "# controlled automatically (the unsymmetric strategy modifies Q, the\n";
    cout << "# others do not). Default: 0.\n";
    cout << "# ----------------------------------------------------------------------\n";


    printVariable_("umfpack_fixq", umfpack_fixq);


    cout << "# Control[UMFPACK_AMD_DENSE]           10 AMD dense row/column parameter\n";
    cout << "# ----------------------------------------------------------------------\n";
    cout << "# rows/columns in A+A' with more than max (16, Control\n";
    cout << "# [UMFPACK_AMD_DENSE] * sqrt (n)) entries (where n = n_row = n_col)\n";
    cout << "# are ignored in the AMD pre-ordering.  Default: 10\n";
    cout << "# ----------------------------------------------------------------------\n";


    printVariable_("umfpack_amd_dense", umfpack_amd_dense);


    cout << "# Control[UMFPACK_SYM_PIVOT_TOLERANCE] 0.001        for diagonal entries\n";
    cout << "# ----------------------------------------------------------------------\n";
    cout << "# If diagonal pivoting is attempted (the symmetric or symmetric-2by2\n";
    cout << "# strategies are used) then this parameter is used to control when the\n";
    cout << "# diagonal entry is selected in a given pivot column. The absoluter\n";
    cout << "# value of the entry must be >= Control [UMFPACK_SYM_PIVOT_TOLERANCE]\n";
    cout << "# times the largest absolute value in the column. A value of zero will\n";
    cout << "# ensure that no off-diagonal pivoting is performed, except that zero\n";
    cout << "# diagonal entries are not selected if there are any off-diagonal\n";
    cout << "# nonzero entries.  If an off-diagonal pivot is selected, an attempt\n";
    cout << "# is made to restore symmetry later on. Suppose A (i,j) is selected,\n";
    cout << "# where i != j.  If column i has not yet been selected as a pivot\n";
    cout << "# column, then the entry A (j,i) is redefined as a \"diagonal\" entry,\n";
    cout << "# except that the tighter tolerance (Control\n";
    cout << "# [UMFPACK_PIVOT_TOLERANCE]) is applied. This strategy has an effect\n";
    cout << "# similar to 2-by-2 pivoting for symmetric indefinite matrices. If a\n";
    cout << "# 2-by-2 block pivot with nonzero structure\n";
    cout << "#\n";
    cout << "#    i j \n";
    cout << "# i: 0 x \n";
    cout << "# j: x 0 \n";
    cout << "#\n";
    cout << "# is selected in a symmetric indefinite factorization method, the\n";
    cout << "# 2-by-2 # block is inverted and a rank-2 update is applied. In\n";
    cout << "# UMFPACK, this # 2-by-2 block would be reordered as \n";
    cout << "#\n";
    cout << "#    j i \n";
    cout << "# i: x 0 \n";
    cout << "# j: 0 x\n";
    cout << "#\n";
    cout << "# In both cases, the symmetry of the Schur complement is preserved.\n";
    cout << "# ----------------------------------------------------------------------\n";


    printVariable_("umfpack_sym_pivot_tolerance", umfpack_sym_pivot_tolerance);


    cout << "# Control[UMFPACK_SCALE]         1 (sum) row scaling (none, sum, or max)\n";
    cout << "# ----------------------------------------------------------------------\n";
    cout << "# Note that the user's input matrix is never modified, only an\n";
    cout << "# internal copy is scaled.  There are three valid settings for this\n";
    cout << "# parameter. If any other value is provided, the default is used.\n";
    cout << "#\n";
    cout << "# UMFPACK_SCALE_NONE: no scaling is performed.  \n";
    cout << "#\n";
    cout << "# UMFPACK_SCALE_SUM: each row of the input matrix A is divided by the\n";
    cout << "# sum of the absolute values of the entries in that row.  The scaled\n";
    cout << "# matrix has an infinity norm of 1.\n";
    cout << "#\n";
    cout << "# UMFPACK_SCALE_MAX: each row of the input matrix A is divided by the\n";
    cout << "# maximum the absolute values of the entries in that row.  In the\n";
    cout << "# scaled matrix the largest entry in each row has a magnitude exactly\n";
    cout << "# equal to 1.  Note that for complex matrices, a cheap approximate\n";
    cout << "# absolute value is used, |a_real| + |a_imag|, instead of the exact\n";
    cout << "# absolute value sqrt ((a_real)^2 + (a_imag)^2).  Scaling is very\n";
    cout << "# important for the \"symmetric\" strategy when diagonal pivoting is\n";
    cout << "# attempted. It also improves the performance of the \"unsymmetric\"\n";
    cout << "# strategy.\n";
    cout << "#\n";
    cout << "# Default: UMFPACK_SCALE_SUM.\n";
    cout << "# ----------------------------------------------------------------------\n";


    printVariable_("umfpack_scale", umfpack_scale);


    cout << "# Control[UMFPACK_FRONT_ALLOC_INIT]  0.5 frontal matrix allocation ratio\n";
    cout << "# ----------------------------------------------------------------------\n";
    cout << "# When UMFPACK starts the factorization of each \"chain\" of frontala\n";
    cout << "# matrices, it allocates a working array to hold the frontal matrices\n";
    cout << "# as they are factorized. The symbolic factorization computes the size\n";
    cout << "# of the largest possible frontal matrix that could occur during the\n";
    cout << "# factorization of each chain.  If Control [UMFPACK_FRONT_ALLOC_INIT]\n";
    cout << "# is >= 0, the following strategy is used. If the AMD ordering was\n";
    cout << "# used, this non-negative parameter is ignored. A front of size\n";
    cout << "# (d+2)*(d+2) is allocated, where d = Info\n";
    cout << "# [UMFPACK_SYMMETRIC_DMAX]. Otherwise, a front of size Control\n";
    cout << "# [UMFPACK_FRONT_ALLOC_INIT] times the largest front possible for this\n";
    cout << "# chain is allocated.  If Control [UMFPACK_FRONT_ALLOC_INIT] is\n";
    cout << "# negative, then a front of size (-Control [UMFPACK_FRONT_ALLOC_INIT])\n";
    cout << "# is allocated (where the size is in terms of the number of numerical\n";
    cout << "# entries). This is done regardless of the ordering method or ordering\n";
    cout << "# strategy used.  \n";
    cout << "#\n";
    cout << "# Default: 0.5.\n";
    cout << "# ----------------------------------------------------------------------\n";


    printVariable_("umfpack_front_alloc_init", umfpack_front_alloc_init);


    cout << "# Control[UMFPACK_DROPTOL]             0                  drop tolerance\n";
    cout << "# ----------------------------------------------------------------------\n";
    cout << "# Entries in L and U with absolute value less than or equal to the\n";
    cout << "# drop tolerance are removed from the data structures (unless leaving\n";
    cout << "# them there reduces memory usage by reducing the space required for\n";
    cout << "# the nonzero pattern of L and U).  Default: 0.0.\n";
    cout << "# ----------------------------------------------------------------------\n";


    printVariable_("umfpack_droptol", umfpack_droptol);


    cout << "# Control[UMFPACK_AGGRESSIVE]          1     (yes) aggressive absorption\n";
    cout << "#                                                 in  AMD and COLAMD\n";
    cout << "# ----------------------------------------------------------------------\n";
    cout << "# If nonzero, aggressive absorption is used in COLAMD and\n";
    cout << "# AMD. \n";
    cout << "#\n";
    cout << "# Default: 1.\n";
    cout << "# ----------------------------------------------------------------------\n";


    printVariable_("umfpack_aggressive", umfpack_aggressive);


#endif


    cout << "\n";
    cout << "\n";

}


void
PardisoParameters::chooseDefaultParameters_()
{
#if defined WITH_PARDISO || defined WITH_PARDISO3
    librarysolver = "PARDISO";
#endif

#ifndef WITH_PARDISO
#ifdef WITH_UMFPACK
    librarysolver = "UMFPACK";
#endif
#endif

#ifndef WITH_PARDISO
#ifndef WITH_UMFPACK
#ifdef WITH_CHOLMOD
    librarysolver = "CHOLMOD";
#endif
#endif
#endif


    verboselevel = 0;
    maxiterations = 100;
    tolerance     = 1e-8;

    pardiso_maxfct = 1;
    pardiso_mnum = 1;
    pardiso_mtype = 11;
    pardiso_msglvl = 1;
    pardiso_iparm[1] = 1;
    pardiso_iparm[2] = 2;
    pardiso_iparm[3] = 1;
    pardiso_iparm[4] = 0;
    pardiso_iparm[5] = 0;
    pardiso_iparm[8] = 0;
    pardiso_iparm[10] = 13;
    pardiso_iparm[11] = 1;
    pardiso_iparm[12] = 0;
    pardiso_iparm[13] = 0;
    pardiso_iparm[18] = 0;
    pardiso_iparm[19] = 0;
    pardiso_iparm[21] = 1;
    pardiso_iparm[28] = 1;
    pardiso_iparm[29] = 0;
    pardiso_iparm[32] = 0;

#ifdef WITH_UMFPACK
    umfpack_prl = 1;
    umfpack_dense_row = 0.2;
    umfpack_dense_col = 0.2;
    umfpack_pivot_tolerance = 0.1;
    umfpack_block_size = 16;
    umfpack_strategy = 1;
    umfpack_alloc_init = 0.7;
    umfpack_irstep = 2;
    umfpack_twobytwo_tolerance = 0.01;
    umfpack_fixq = 0;
    umfpack_amd_dense = 10;
    umfpack_sym_pivot_tolerance = 0.001;
    umfpack_scale = 1;
    umfpack_front_alloc_init = 0.5;
    umfpack_droptol = 0;
    umfpack_aggressive = 1;
#endif

}


void
PardisoParameters::init(string datafile)
{
    // first call base class init to read all the data and place them
    // in the private objects of type vector<string>
    ReadData::init(datafile);

    initializeVariable_("librarysolver", librarysolver);
//    initializeVariable_("verboselevel", verboselevel);
//    initializeVariable_("maxiterations", maxiterations);
//    initializeVariable_("tolerance", tolerance);



        initializeVariable_("pardiso_maxfct", pardiso_maxfct);
        initializeVariable_("pardiso_mnum", pardiso_mnum);
        initializeVariable_("pardiso_mtype", pardiso_mtype);
        initializeVariable_("pardiso_msglvl", pardiso_msglvl);
        initializeVariable_("pardiso_iparm[1]", pardiso_iparm[1]);
        initializeVariable_("pardiso_iparm[2]", pardiso_iparm[2]);
        initializeVariable_("pardiso_iparm[3]", pardiso_iparm[3]);
        initializeVariable_("pardiso_iparm[4]", pardiso_iparm[4]);
        initializeVariable_("pardiso_iparm[5]", pardiso_iparm[5]);
        initializeVariable_("pardiso_iparm[6]", pardiso_iparm[6]);
        initializeVariable_("pardiso_iparm[8]", pardiso_iparm[8]);
        initializeVariable_("pardiso_iparm[10]", pardiso_iparm[10]);
        initializeVariable_("pardiso_iparm[11]", pardiso_iparm[11]);
        initializeVariable_("pardiso_iparm[12]", pardiso_iparm[12]);
        initializeVariable_("pardiso_iparm[13]", pardiso_iparm[13]);
        initializeVariable_("pardiso_iparm[18]", pardiso_iparm[18]);
        initializeVariable_("pardiso_iparm[19]", pardiso_iparm[19]);
        initializeVariable_("pardiso_iparm[21]", pardiso_iparm[21]);
        initializeVariable_("pardiso_iparm[28]", pardiso_iparm[28]);
        initializeVariable_("pardiso_iparm[29]", pardiso_iparm[29]);
        initializeVariable_("pardiso_iparm[32]", pardiso_iparm[32]);

#ifdef WITH_UMFPACK
    if (librarysolver == "UMFPACK" || librarysolver == "SCHUR_UMFPACK")
    {
        initializeVariable_("umfpack_prl", umfpack_prl);
        initializeVariable_("umfpack_dense_row", umfpack_dense_row);
        initializeVariable_("umfpack_dense_col", umfpack_dense_col);
        initializeVariable_("umfpack_pivot_tolerance", umfpack_pivot_tolerance);
        initializeVariable_("umfpack_block_size", umfpack_block_size);
        initializeVariable_("umfpack_strategy", umfpack_strategy);
        initializeVariable_("umfpack_alloc_init", umfpack_alloc_init);
        initializeVariable_("umfpack_irstep", umfpack_irstep);
        initializeVariable_("umfpack_twobytwo_tolerance", umfpack_twobytwo_tolerance);
        initializeVariable_("umfpack_fixq", umfpack_fixq);
        initializeVariable_("umfpack_amd_dense", umfpack_amd_dense);
        initializeVariable_("umfpack_sym_pivot_tolerance", umfpack_sym_pivot_tolerance);
        initializeVariable_("umfpack_scale", umfpack_scale);
        initializeVariable_("umfpack_front_alloc_init", umfpack_front_alloc_init);
        initializeVariable_("umfpack_droptol", umfpack_droptol);
        initializeVariable_("umfpack_aggressive", umfpack_aggressive);
    }
#endif

}

void
PardisoParameters::init(fstream& fin)
{
    // first call base class init to read all the data and place them
    // in the private objects of type vector<string>
    ReadData::init(fin);

    initializeVariable_("librarysolver", librarysolver);
    initializeVariable_("verboselevel", verboselevel);
    initializeVariable_("maxiterations", maxiterations);
    initializeVariable_("tolerance", tolerance);


    initializeVariable_("pardiso_maxfct", pardiso_maxfct);
    initializeVariable_("pardiso_mnum", pardiso_mnum);
    initializeVariable_("pardiso_mtype", pardiso_mtype);
    initializeVariable_("pardiso_msglvl", pardiso_msglvl);
    initializeVariable_("pardiso_iparm[1]", pardiso_iparm[1]);
    initializeVariable_("pardiso_iparm[2]", pardiso_iparm[2]);
    initializeVariable_("pardiso_iparm[3]", pardiso_iparm[3]);
    initializeVariable_("pardiso_iparm[4]", pardiso_iparm[4]);
    initializeVariable_("pardiso_iparm[5]", pardiso_iparm[5]);
    initializeVariable_("pardiso_iparm[6]", pardiso_iparm[6]);
    initializeVariable_("pardiso_iparm[8]", pardiso_iparm[8]);
    initializeVariable_("pardiso_iparm[10]", pardiso_iparm[10]);
    initializeVariable_("pardiso_iparm[11]", pardiso_iparm[11]);
    initializeVariable_("pardiso_iparm[12]", pardiso_iparm[12]);
    initializeVariable_("pardiso_iparm[13]", pardiso_iparm[13]);
    initializeVariable_("pardiso_iparm[18]", pardiso_iparm[18]);
    initializeVariable_("pardiso_iparm[19]", pardiso_iparm[19]);
    initializeVariable_("pardiso_iparm[21]", pardiso_iparm[21]);
    initializeVariable_("pardiso_iparm[28]", pardiso_iparm[28]);
    initializeVariable_("pardiso_iparm[29]", pardiso_iparm[29]);
    initializeVariable_("pardiso_iparm[32]", pardiso_iparm[32]);

#ifdef WITH_UMFPACK
    initializeVariable_("umfpack_prl", umfpack_prl);
    initializeVariable_("umfpack_dense_row", umfpack_dense_row);
    initializeVariable_("umfpack_dense_col", umfpack_dense_col);
    initializeVariable_("umfpack_pivot_tolerance", umfpack_pivot_tolerance);
    initializeVariable_("umfpack_block_size", umfpack_block_size);
    initializeVariable_("umfpack_strategy", umfpack_strategy);
    initializeVariable_("umfpack_alloc_init", umfpack_alloc_init);
    initializeVariable_("umfpack_irstep", umfpack_irstep);
    initializeVariable_("umfpack_twobytwo_tolerance", umfpack_twobytwo_tolerance);
    initializeVariable_("umfpack_fixq", umfpack_fixq);
    initializeVariable_("umfpack_amd_dense", umfpack_amd_dense);
    initializeVariable_("umfpack_sym_pivot_tolerance", umfpack_sym_pivot_tolerance);
    initializeVariable_("umfpack_scale", umfpack_scale);
    initializeVariable_("umfpack_front_alloc_init", umfpack_front_alloc_init);
    initializeVariable_("umfpack_droptol", umfpack_droptol);
    initializeVariable_("umfpack_aggressive", umfpack_aggressive);
#endif

}


