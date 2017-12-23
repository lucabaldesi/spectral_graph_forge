This code release consists of two classes (i) the "Generation" and (ii) the "Estimation" class. The following functions are exposed in each class.

Estimation.py
------------
* load_graph(fname)         ## loads graph (accepts edges.gz &  .mat formats)
* calcfull_CCK()            ## calculates c(k) distribution  (assumes "load_graph" has been called)
* calcfull_JDD()            ## calculates JDD distribution (assumes "load_graph" has been called)
* sample('rw', p_sample)    ## collects a random walk sample (assumes "load_graph" has been called)
* estimate_CCK()            ## estimates c(k) distribution (assumes "load_graph", "sample" have been called)
* estimate_JDD()            ## estimates JDD distribution (assumes "load_graph", "sample" have been called)
* estimation_summary()      ## provides summary of either estimation or calculation of JDD and CCK.

* get_KTRI('full')          ## returns full KTRI (assumes "load_graph", "calcfull_CCK" have been called)
* get_KTRI('estimate')      ## returns estimated KTRI (assumes "load_graph", "sample", "estimate_CCK" have been called)
* get_JDD('full')           ## returns full JDD (assumes "load_graph", "calcfull_JDD" have been called)
* get_JDD('realizable')     ## returns realizable JDD (assumes "load_graph", "sample", "estimate_JDD" have been called)

KTRI is the non-normalized version of c(k). KTRI contains the number of triangles connected to nodes of degree k. 




Generation.py
------------
* set_JDD(  jdd_input)  ## sets JDD  (from get_JDD of Estimation)
* set_KTRI( jdd_input ) ## sets KTRI (from get_KTRI of Estimation)
        
* construct_triangles_2K()  ## construct  simple 2K with high clustering  (our algorithm) - assumes "set_JDD" has been called
* construct_simple_2K()     ## construct  simple 2K  (Sandia labs version) - assumes "set_JDD" has been called
* construct_dkseries_2K()   ## construct  2K with potential multigraph edges (CAIDA version) - assumes "set_JDD" has been called


* mcmc_improved_2_5_K(error_threshold=0.03) ## do improved MCMC to reach  2.5K  - assumes "set_JDD", "set_KTRI" have been called and  one of the three 2K constructions has been   completed.
* mcmc_random_2_5_K(error_threshold=0.03) ## do random MCMC to reach 2.5K  - assumes "set_JDD", "set_KTRI" have been called and  one of the three 2K constructions has been   completed.
* save_graphs(filename)  ## save constructed graph to a python pickle file.


        
examples.py
-----------               
This file contains examples that demonstrate these functions.



Our paper "2.5K-Graphs: from Sampling to Generation" accessible at http://www.minasgjoka.com/papers/2.5K_Graphs.pdf  provides a description of the estimation and construction functions.