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
    #print("\nenvvar: "+str(envvar))
    #print("\ntest_list: "+str(TEST_LIST))

    #
    # Storing the value of each required env var
    #
    envvar_content=[]
    n=len(envvar)
    i=0
    while i < n:
      #print(os.getenv(envvar[i]))
      envvar_content.append(os.getenv(envvar[i])) 
      #print("i="+str(i)+"\t   => "+str(envvar[i])+"\t   => "+str(os.getenv(envvar[i]))+"\t   => "+str(envvar_content))
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
       #print("j="+str(j)+"     "+str(envvar_toundef))
    
       for var in envvar_toundef:
         #print("getting 1 "+var) 
         #print(str(os.getenv(var)))
         #print("removing "+var) 
         del os.environ[var] 
         #print("getting 2 "+var) 
         #print(str(os.getenv(var)))

       #print("**qui si esegue il nosetests")
       result = nose.run(argv=[sys.argv[0],TEST_LIST,'-v','-s'])
       #
       # Checking exit status 
       #  
       print("result = "+str(result))
       #if result != "False":
       #  print(" Test case removing "+str(j)+" enviroment variables failed... ok") 
       #else:
       #  print(" Test case removing "+str(j)+" enviroment variables passed... ")


       #
       # Setting again the removed envvar (to let the test failing) by getting the corresponding value from envvar_content
       #
       for var in envvar_toundef:
         i=0
         while i < n:
           if var == envvar[i]: 
             #print("i= "+str(i)+", var= "+var+", envvar[i]= "+str(envvar[i])+", envvar_content[i]= "+str(envvar_content[i]))
             os.environ[var]=str(envvar_content[i])
             #print(str(os.getenv(var)))
           i=i+1
 


    #print("Getting here the value of each var in envvar")
    #for v in envvar:
    #   print(v+"\t\t->"+str(os.getenv(v)))
    #
    # Running now the test successfully
    #
    #nose.run(argv=[sys.argv[0],TEST_LIST,'-v'])


