import unittest
import nose

import os
import sys
import random

from gridtools import CUDA_ENVVARS
from gridtools import TEST_LIST

class EnvVarMissing(unittest.TestCase):

  def test_envvarmissing(self):

    envvar=CUDA_ENVVARS

    #
    # Storing the value of each required env var
    #
    envvar_content=[]
    n=len(envvar)
    i=0
    while i < n:
      envvar_content.append(os.getenv(envvar[i])) 
      i=i+1


    #
    # Defining subesers in envvar randomly
    #   j starts from 1 because for j=0, envvar_toundef is empty and the for on var is not done 
    #   (i.e. the gridtools tests are executed successfully)
    #
    for j in range(1,n+1):
       # 
       # Defining a subset of envvar (without repetitions) 
       # 
       print("\n\n*** Removing "+str(j)+" enviroment variables")
       envvar_toundef=random.sample(envvar,j)
    
       for var in envvar_toundef:
         del os.environ[var] 

       result = nose.run(argv=[sys.argv[0],TEST_LIST,'-v','-s'])
       #
       # Checking exit status 
       #  
       if result != "False":
         print("*** Test case removing "+str(j)+" enviroment variables failed... ok") 
       else:
         print("*** Test case removing "+str(j)+" enviroment variables passed... ")


       #
       # Setting again the removed envvar by getting the corresponding value from envvar_content
       #
       for var in envvar_toundef:
         i=0
         while i < n:
           if var == envvar[i]: 
             os.environ[var]=str(envvar_content[i])
           i=i+1
 
