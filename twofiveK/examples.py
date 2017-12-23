'''
Examples of usage for classes Estimation and Generation.


Changelog
****************

v5: Sept 2013
Added Estimation Summary

v4: Mar 2013
Added graceful interruption with CTRL+C
Added information about package requirements 

v1: Aug 2012
initial release

'''

__author__ = """Minas Gjoka"""

import signal, sys, os

def handler(signum, frame_unused):
  if (signum ==signal.SIGINT) or (signum ==signal.SIGTERM) :	
    if "mygen" in globals() and "fname" in globals():
      print "Process interrupted. Attempting to dump partially constructed 2K and 2.5K graphs."
      mygen.save_graphs('%s_interrupted' % fname)
    sys.exit()    

if __name__ == "__main__":

  print "Author: Minas Gjoka"
  print "Demonstration of 2.5K algorithms"
  print "Requires Python packages 'networkx', 'numpy', 'scipy'"
  print "Tested with Python versions 2.6.x and 2.7.x"
  try:
    ## do graceful interruption (i.e. CTRL+C)
    signal.signal(signal.SIGINT, handler)
    signal.signal(signal.SIGTERM, handler)
    print "CTRL+C during the 'double edge swap phase' will dump the partially targetted 2.5K graph"
  except:
    pass    
  print "-------------------------------------------"
  print "Press 'Enter' to start the graph construction"
  raw_input()


  from Estimation import Estimation
  from Generation import Generation	

  '''
    You can download below topologies at 
    http://www.minasgjoka.com/2.5K/graphs
    '''
  #fname = "UCSD34.mat"    
  #fname = "Harvard1.mat"
  #fname = "Facebook-New-Orleans.edges.gz"
  #fname = 'soc-Epinions1.edges.gz'
  #fname = "email-Enron.edges.gz"
  #fname = "as-caida20071105.edges.gz"        
  #fname = "out.ca-AstroPh.edges.gz"
  fname = "powerlaw_cluster"   # from networkx.generators.powerlaw_cluster_graph ; n=5000

  #fname = "web-NotreDame.edges.gz"
  #fname = "web-Google.edges.gz"
  #fname = "wiki-Talk.edges.gz"
  run_case = 1
  error_threshold = 0.05

  ###### Full graph - 2K with triangles + Improved MCMC 
  if run_case == 1:
    myest = Estimation()
    myest.load_graph(fname)

    myest.calcfull_CCK()
    myest.calcfull_JDD()
    myest.estimation_summary()
    
    mygen = Generation()
    mygen.set_JDD( myest.get_JDD('full') )
    mygen.set_KTRI( myest.get_KTRI('full') ) 

    mygen.construct_triangles_2K()
    mygen.mcmc_improved_2_5_K(error_threshold=error_threshold)
    mygen.save_graphs('%s_2KT+ImpMCMC_Full' % fname)
  #######################################################
  ###### Full graph - 2K simple +  MCMC  
  elif run_case == 2:
    myest = Estimation()
    myest.load_graph(fname)
    
    myest.calcfull_CCK()
    myest.calcfull_JDD()
    myest.estimation_summary()

    mygen = Generation()
    mygen.set_JDD( myest.get_JDD('full') )
    mygen.set_KTRI( myest.get_KTRI('full') ) 

    mygen.construct_simple_2K()
    mygen.mcmc_random_2_5_K(error_threshold=error_threshold)
    mygen.save_graphs('%s_2Ksimple+MCMC_Full' % fname)
  #######################################################
  ###### 30% sample - 2K with triangles + Improved MCMC 
  elif run_case == 3:
    p_sample = 0.4
    myest = Estimation()
    myest.load_graph(fname)
    
    myest.sample('rw', p_sample)
    myest.estimate_JDD()
    myest.estimate_CCK()
    myest.estimation_summary()

    mygen = Generation()
    mygen.set_JDD( myest.get_JDD('realizable') )
    mygen.set_KTRI( myest.get_KTRI('estimate') ) 

    mygen.construct_triangles_2K()
    mygen.mcmc_improved_2_5_K(error_threshold=error_threshold)
    mygen.save_graphs('%s_2KT+ImpMCMC_%.2fsample' % (fname, p_sample))
  #######################################################
  ###### 30% sample - 2K simple +  MCMC  
  elif run_case == 4:
    p_sample = 0.4
    myest = Estimation()
    myest.load_graph(fname)
    myest.sample('rw', p_sample)
    
    myest.estimate_JDD()
    myest.estimate_CCK()
    myest.estimation_summary()

    mygen = Generation()
    mygen.set_JDD( myest.get_JDD('realizable') )
    mygen.set_KTRI( myest.get_KTRI('estimate') ) 

    mygen.construct_simple_2K()
    mygen.mcmc_random_2_5_K(error_threshold=error_threshold)
    mygen.save_graphs('%s_2Ksimple+MCMC_%.2fsample' % (fname, p_sample))        

