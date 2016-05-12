import unittest
import logging

import numpy as np

from gridtools.stencil  import MultiStageStencil


class GameOfLife (MultiStageStencil):
    """
    # Tests various parts of the if-statement, notably:
    # 1) Single-clause conditional
    # 2) Multi-clause conditional connected by "and"
    # 3) Single-statement body
    # 4) Single-statement else
    # 5) Else-if block
    """
    def __init__ (self, domain):
        super ( ).__init__ ( )
        self.counter = np.zeros (domain)
        self.set_halo ( (1,1,1,1) )


    def kernel (self, out_X, in_X):
        for p in self.get_interior_points (out_X):

            self.counter[p] = out_X[p + (1,0,0)]  + out_X[p + (1,1,0)]   + \
                              out_X[p + (0,1,0)]  + out_X[p + (-1,1,0)]  + \
                              out_X[p + (-1,0,0)] + out_X[p + (-1,-1,0)] + \
                              out_X[p + (0,-1,0)] + out_X[p + (1,-1,0)]

            if out_X[p] == 1.0 and self.counter[p] == 2:
                out_X[p] = 0.0
            elif self.counter[p] == 3:
                out_X[p] = 1.0
            else:
                out_X[p] = 0.0



class AdditionalIfStatement (MultiStageStencil):
    """
    # Additional tests of the if-statement, notably:
    # 1) Multi-clause conditional connected by "and" and "or"
    # 2) Multi-clause conditional connected by "and" and "or" with parenthesis
    # 3) Multi-statement body
    # 4) Multi-statement else
    # 5) Not operator used within the conditional
    # 6) Uses the following relational operators: >, <, <=, >=, ==, !=
    # 7) Tests an if-statment without any else clause
    """
    def __init__ (self, domain):
        super ( ).__init__ ( )
        self.counter = np.zeros (domain)
        self.set_halo ( (1,1,1,1) )


    def kernel (self, out_X, in_X):
        for p in self.get_interior_points (out_X):
            self.counter[p] = out_X[p + (1,0,0)]

            if (out_X[p] != 1.0 and self.counter[p] == 2 and out_X[p + (1,0,0)] > 0.0) or self.counter[p] < 10:
                out_X[p] = 0.0
            if out_X[p] == 2.0 and (self.counter[p] <= 10 or out_X[p] < 3.2):
                out_X[p] = 1.0

            if self.counter[p] == 1.0:
                out_X[p] = 0.0
            elif self.counter[p] >= 3:
                out_X[p] = 1.0
                out_X[p + (1,0,0)] = 1.0
            else:
                out_X[p] = 0.0
                out_X[p] = 1.0
                out_X[p] = 2.0

            if not out_X[p]:
                out_X[p] = 0.5



class IfStatementOpIsFailure (MultiStageStencil):
    """
    Tests that use of 'is' operator currently raises an error.
    """
    def __init__ (self, domain):
        super ( ).__init__ ( )
        self.set_halo ( (1,1,1,1) )


    def kernel (self, out_X):
        for p in self.get_interior_points (out_X):
            if out_X[p] is out_X[p]:
                out_X[p] = 0.0



class IfStatementOpIsNotFailure (MultiStageStencil):
    """
    Tests that use of 'is not' operator currently raises an error.
    """
    def __init__ (self, domain):
        super ( ).__init__ ( )
        self.set_halo ( (1,1,1,1) )


    def kernel (self, out_X):
        for p in self.get_interior_points (out_X):
            if out_X[p] is not out_X[p]:
                out_X[p] = 0.0



class IfStatementOpNotInFailure (MultiStageStencil):
    """
    Tests that use of 'not in' operator currently raises an error.
    """
    def __init__ (self, domain):
        super ( ).__init__ ( )
        self.set_halo ( (1,1,1,1) )


    def kernel (self, out_X):
        for p in self.get_interior_points (out_X):
            if out_X[p] not in out_X[p]:
                out_X[p] = 0.0



class IfStatementOpInFailure (MultiStageStencil):
    """
    Tests that use of 'in' operator currently raises an error.
    """
    def __init__ (self, domain):
        super ( ).__init__ ( )
        self.set_halo ( (1,1,1,1) )


    def kernel (self, out_X):
        for p in self.get_interior_points (out_X):
            if out_X[p] in out_X[p]:
                out_X[p] = 0.0



class IfStatementTest (unittest.TestCase):
    """
    A test case for the If-statement related stencils defined above.-
    """
    def _run (self, stencil):
        kwargs = dict ( )
        for p in self.params:
            kwargs[p] = getattr (self, p)
        stencil.run (**kwargs)

    def setUp (self):
        super ( ).setUp ( )
        logging.basicConfig (level=logging.INFO)
        self.domain = (64, 64, 32)
        self.params = ('out_X','in_X')


        self.in_X = np.random.random_integers (10, 
                                                 size=self.domain)
        self.in_X = self.in_X.astype (np.float64)

        self.out_X = np.copy (self.in_X)

        self.stencil  = GameOfLife                 (self.domain)
        self.stencil2 = AdditionalIfStatement      (self.domain)
        self.stencil3 = IfStatementOpIsFailure     (self.domain)
        self.stencil4 = IfStatementOpIsNotFailure  (self.domain)
        self.stencil5 = IfStatementOpNotInFailure  (self.domain)
        self.stencil6 = IfStatementOpInFailure     (self.domain)


    def test_game_of_life (self):
        from tests.test_stencils import CopyTest
        ct         = CopyTest ( )
        ct.stencil = self.stencil
        ct.domain  = self.domain
        ct.params  = self.params
        ct.out_X   = self.out_X
        ct.in_X    = self.in_X
        ct.test_compare_python_cpp_and_cuda_results ( )


    def test_additional_if_statement_tests (self):
        from tests.test_stencils import CopyTest
        ct         = CopyTest ( )
        ct.stencil = self.stencil2
        ct.domain  = self.domain
        ct.params  = self.params
        ct.out_X   = self.out_X
        ct.in_X    = self.in_X
        ct.test_compare_python_cpp_and_cuda_results ( )

#    def test_op_is_raises_error (self):
#        with self.assertRaises (NotImplementedError):
#            self.stencil3.backend = 'c++'
#            self._run (self.stencil3)
#
#
#    def test_op_is_not_raises_error (self):
#        with self.assertRaises (NotImplementedError):
#            self.stencil4.backend = 'c++'
#            self._run (self.stencil4)
#
#
#    def test_op_not_in_raises_error (self):
#        with self.assertRaises (NotImplementedError):
#            self.stencil5.backend = 'c++'
#            self._run (self.stencil5)
#
#
#    def test_op_in_raises_error (self):
#        with self.assertRaises (NotImplementedError):
#            self.stencil6.backend = 'c++'
#            self._run (self.stencil6)
