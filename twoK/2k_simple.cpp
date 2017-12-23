#include "2k_simple.h"


class GraphUndir {
    mapi_seti node_neighbs ;
    
    public:
    veci & nodes() {
        veci * result = new veci;
        tr(node_neighbs,it){
            result->push_back( it->first );
        }
        return *result;
    }
    
    seti &  neighbors(int v){
        return node_neighbs[v];
    }
    
    void add_edge(int v, int w){
        node_neighbs[v].insert(w);
        node_neighbs[w].insert(v);
    }
        
    void remove_edge(int v, int w){
        node_neighbs[v].erase(w);
        node_neighbs[w].erase(v);
    }
        
    bool has_edge(int v, int w) {
        mapi_seti::iterator iter_node = node_neighbs.find(v);
        
        if (iter_node != node_neighbs.end()) {
           seti::iterator iter_neighb = iter_node->second.find(w) ;
            if (iter_neighb != iter_node->second.end()) {
                return true;
            }
        }
        return false;
    }
    
    int size() {
        return node_neighbs.size();
    }
};
 
 
bool is_valid_joint_degree(mapi_mapii & nkk){
    /* Checks whether the given joint degree (nkk) is realizable 
        as a simple graph by evaluating five necessary and sufficient conditions.
        

    Parameters
    ----------
    nkk :  map of map of integers
        joint degree. for nodes of degree k (first level of map) and
        nodes of degree l (second level of map) describes the number of edges        
    
    Returns
    -------
    boolean
        returns true if given nkk is realizable, else returns false.

    
    References
    ----------
    [1] M. Gjoka, M. Kurant, A. Markopoulou, "2.5K Graphs: from Sampling to Generation",
    IEEE Infocom, 2013.
    */

    // list of five conditions that a nkk needs to satisfy for simple graph realizability
    // cond. 1:  nkk[k][l]  is integer for all k,l
    // cond. 2:  sum(nkk[k])/k = number of nodes with degree k, is an integer
    // cond. 3,4: number of edges between k and l cannot exceed maximum possible number of edges 
    // cond. 5: nkk[k][k] is an even integer i.e. stubs are counted twice for equal degree pairs.
    // cond. 5: this is an assumption that the method joint_degree_model expects to be true.
    
    
    
    // condition 1 satisfied by the definition of nkk as a map of map integers 
    
    
    // compute "degree map" nk from joint degree nkk
    mapii nk;
    int s, deg ;
    double k_size, intpart;
    tr(nkk,it1) {
        s = 0;
        deg = it1->first;        
        
        tr(it1->second, it2) {
            s += it2->second;         
        }
        k_size = double(s)/double(deg);
        
        if ( modf(k_size, & intpart) != 0) {   // conndition 2
            cout << "Violation of condition 2" <<endl;
            return false;
        }
        
        nk[deg] = (int) k_size ;        
    }
              
    int k, l;
    tr(nkk,it1) {
        k = it1->first;
        
        tr(it1->second, it2) {
            l = it2->first;
            
            // make nk values double to avoid integer overflow
            if ((k!=l) && (nkk[k][l] > double(nk[k]) * double(nk[l]) ) ){ // condition 3
                cout << "Violation of condition 3" <<endl;
                return false;
            } else if (k==l) {
                if ( nkk[k][k] > double(nk[k])*double(nk[k]-1) ) { // condition 4
                    cout << "Violation of condition 4" <<endl;
                    return false;
                }
                if (nkk[k][k] % 2 != 0) { // condition 5
                    cout << "Violation of condition 5" <<endl;
                    return false;
                }
            }
        }
    }
            
        
    // if all five above conditions have been satisfied then the input nkk is 
    //realizable as a simple graph.            
    return true;
}

void neighbor_switch(GraphUndir & G, int & w, seti & unsat, mapii & h_node_residual, int avoid_node_id=NULL) { 

    /* neighbor_switch  releases one free stub for node w. First, it selects node w_prime
    that (1) has the same degree as w and (2) is unsaturated. Then, it selects node t,
    a neighbor of w, that is not connected to w_prime and does an edge swap i.e.
    removes (w,t) and adds (w_prime,t). Gjoka et. al. [1] prove that if (1) and (2) 
    are true then such an edge swap is always possible.     
        

    Parameters
    ----------
    G :  undirected graph 
        graph within which the edge swap will take place
    w : integer
        node id for which we need to perform a neighbor switch 
    unsat: set of integers
       set of node ids that have the same degree as w and are unsaturated
    h_node_residual: map of integers
        for a given node, keeps track of the remaining stubs to be added.
    avoid_node_id: integer
        node id to avoid when selecting w_prime. only used for a rare edge case.
    
    
    References
    ----------
    [1] M. Gjoka, B. Tillman, A. Markopoulou, "Construction of Simple Graphs 
       with a Target Joint Degree Matrix and Beyond", IEEE Infocom, 2015.
    
    */
    
    int w_prime;
    int t;
    
    if ( (avoid_node_id==NULL) || (h_node_residual[avoid_node_id]>1) ){
        // select node w_prime that has the same degree as w and is unsatured        
        tr(unsat,it){
            w_prime = *it;
            break;
        }  
    }
    else {
         // assume that inside method joint_degree_model the node pair (v,w) is a candidate
         // for connection (v=avoid_node_id). if neighbor_switch is called for node w inside method 
         // joint_degree_model and (1) candidate nodes v=avoid_node_id and w have the same degree i.e.
         // degree(v)=degree(w) and (2) node v=avoid_node_id is a potential candidate for w_prime
         // but has only  one stub left i.e. h_node_residual[v]==1, then prevent v from
         // being selected as w_prime. This is a rare edge case. 
    
        tr(unsat,it){
            w_prime = *it;
            if (w_prime != avoid_node_id){
                break;
            }
        }          
    }

    
    
    seti * wprime_neighbors = & G.neighbors(w_prime);
    
    // select node t, a neighbor of w, that is not connected to w_prime
    tr( (G.neighbors(w)), it){
        if(wprime_neighbors->find(*it)==wprime_neighbors->end() && (*it!=w_prime) ) { // not found
            t = *it;
            break;
        }
    }   
    
    // removes (w,t), add (w_prime,t)  and update data structures
    G.remove_edge(w, t) ;
    G.add_edge(w_prime, t) ;
    h_node_residual[w] += 1 ;                           
    h_node_residual[w_prime] -= 1 ;
    if (h_node_residual[w_prime] == 0) {
        unsat.erase(w_prime);
    }
    
    
}
        
void joint_degree_model(mapi_mapii & nkk, GraphUndir & G) {
    /* Return a random simple graph with the given joint degree (nkk).

    Parameters
    ----------
    nkk :  map of map of integers
        joint degree. for nodes of degree k (first level of map) and
        nodes of degree l (second level of map) describes the number of edges        
    G : Graph
        Graph  G will have the specified joint degree upon completion

    Notes
    -----
    todo: add description

    References
    ----------
    [1] M. Gjoka, B. Tillman, A. Markopoulou, "Construction of Simple Graphs 
       with a Target Joint Degree Matrix and Beyond", IEEE Infocom, 2015.

    */
    
    cout << "joint_degree_model" << endl;
    
    if (!is_valid_joint_degree(nkk)) {
        cout << "Input joint degree (nkk) not realizable as a simple graph.";
        return;
    }
    

    
    // compute "degree map" nk from joint degree nkk
    mapii nk;

    tr(nkk,it1) {
        int s = 0;
        int deg = it1->first;        
        
        tr(it1->second, it2) {
            s += it2->second;         
        }
        nk[deg] = s/deg;
        
    }
    
        
    //for a given degree group, keep the list of all node ids 
    mapi_veci h_degree_nodelist ;
    
    // for a given node, keep track of the remaining stubs to be added. 
    mapii h_node_residual ;
    
    // populate h_degree_nodelist and h_node_residual
    int nodeid = 0 ;
    tr(nk, it1){
        int degree = it1->first;
        int numNodes = it1->second;

        for (int v=nodeid; v<nodeid+numNodes; v++) {
            h_degree_nodelist[degree].push_back(v);
            h_node_residual[v] = degree;
        }       
        nodeid += numNodes;
        
    }
    
    // iterate over every degree pair and add the number edges given for each pair
    int E=0;
    int n_switches=0;
    int k, l, k_size, l_size;
    int v, w, n_edges_add;
    tr(nkk,it1) {
        k = it1->first;
        
        tr(it1->second, it2) {
            l = it2->first;
            
            // n_edges_add is the number of edges to add for the degree pair (k,l)  
            n_edges_add = it2->second;
            
             // degree pair (k,l)
            if ((n_edges_add > 0) && (k>=l)) {
                
                // k_size and l_size are the number of nodes with degree k and l
                k_size = nk[k];
                l_size = nk[l];
                
                // k_unsat and l_unsat consist of nodes of degree k and l that are unsaturated
                // i.e. those nodes that have at least one available stub
                seti k_unsat;
                seti l_unsat;                
                                
                veci * k_nodes =  & h_degree_nodelist[k];
                veci * l_nodes =  & h_degree_nodelist[l];
                                                                
                tr((*k_nodes),v){
                    if (h_node_residual[*v]>0) {
                        k_unsat.insert(*v);
                    }
                }
                        
                if (k!=l) {                                        
                    tr((*l_nodes),w){
                        if (h_node_residual[*w]>0) {
                            l_unsat.insert(*w);
                        }
                    }
                }                    
                else {
                    n_edges_add = n_edges_add/2 ;
                }            
                                
                while (n_edges_add > 0) {
                                        
                    v = (*k_nodes)[ rand() % k_size];
                    w = (*l_nodes)[ rand() % l_size];


                    // if nodes v and w are disconnected then attempt to connect
                    if ( (!G.has_edge(v,w)) && (v!=w) ) {
                        
                        // if node v has no free stubs then do neighbor switch
                        if (h_node_residual[v]==0) {                         
                            neighbor_switch(G, v, k_unsat, h_node_residual) ;   
                            n_switches+=1;
                        }
                        
                            
                        // if node w has no free stubs then do neighbor switch
                        if (h_node_residual[w]==0) {                   
                            if (k!=l) {
                                neighbor_switch(G, w, l_unsat, h_node_residual);            
                            }
                            else {
                                neighbor_switch(G, w, k_unsat, h_node_residual, v) ;               
                            }
                            n_switches+=1;
                        }
                       
                            
                        // add edges (v,w) and update data structures
                        E += 1;
                        G.add_edge(v,w) ;
                        
                        h_node_residual[v] -= 1 ;
                        h_node_residual[w] -= 1 ; 
                        n_edges_add -= 1 ;
                            
                        if (h_node_residual[v] == 0) {
                            k_unsat.erase(v) ;
                        }
                        if (h_node_residual[w] == 0) {
                            if (k!=l) {
                                l_unsat.erase(w) ;
                            }
                            else {
                                k_unsat.erase(w) ;
                            }
                        }
                    }   
                }                      

            }
        }
    }
    cout << "#Switches:" << n_switches  << endl;
    cout << "#Edges:" << E  << endl;

    return;
    
}

void load_nkk(char * fname, mapi_mapii & nkk){
    ifstream infile( fname );
     
    cout << "Loading file " << fname ;
    // expecting one line per entry of Joint Degree Matrix
    // each entry k, l, nkk[k][l] means: for degree k and degree l there
    // are nkk[k][l] edges in the graph 
    while (infile) {
        string s;
        if (!getline( infile, s )) break;

        istringstream ss( s );
        veci record;
        int i=0;
        while (ss)
        {
          string s;
          if (!getline( ss, s, ',' )) break;
          int val = atoi( s.c_str());
          record.push_back( val);
        }
        if (record.size()==3) {
            nkk[ record[0] ][ record[1] ] = record[2];
        } else{
            cout << "Error loading this row" << endl;
        }
    }
    cout << "  Done. " <<endl;
}

void write_graph(char * fname, GraphUndir & G ) {
    ofstream outfile( fname );
     
    cout << "Writing file " << fname << " ." << endl;
    
    int E = 0;
    veci nodes = G.nodes();
    tr(nodes, v) {
        seti * neigbs = & G.neighbors(*v);
        tr( (*neigbs) , w) {
                if (*v == *w) {
                    cout << "Error: self-loop " << *v << "," << *w << endl;
                }
                outfile  << *v << "," << *w << endl;
                E += 1;
        }
    }
    cout << E << " edges. Done. " <<endl;
    
}

int main(int argc, char * argv[]){
    // initialize random instance
    srand( time(NULL));

    
    char * fname ;
    if (argc>1) {
        fname = argv[1];
    } else {
        cout << "Error: expecting .nkk filename as parameter " << endl;
        return 0 ;
    }    
    
    mapi_mapii nkk;   
    
    //Load NKK
    load_nkk(fname, nkk);    
    
    cout << "Runnning construction\n" ;
    struct timeval tp1, tp2;
    gettimeofday(&tp1,NULL);

    // Construct graph
    GraphUndir g ;
    joint_degree_model(nkk, g);
    cout << "#Nodes:" << g.size() <<endl;
    
    gettimeofday(&tp2,NULL);
    double runtime = double(((tp2.tv_sec - tp1.tv_sec)*1000000) + 
                            (tp2.tv_usec - tp1.tv_usec))/double(1000000);
                            
    cout <<  "Time:" << fixed << std::setprecision(3) << runtime <<endl;
    
    // Write graph
    write_graph("generated.graph", g);
    cout << "Constructed graph 'generated.graph' dumped in edgelist format" << endl;

    return 0;
}
