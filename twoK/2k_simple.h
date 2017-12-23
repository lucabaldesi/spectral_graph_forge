#include <iostream>
#include <vector>
#include <cstdlib> 

#include <stdio.h>
#include <math.h>


#include <fstream>
#include <sstream>
#include <string>


#include <sys/time.h>
#include <iomanip>


using namespace std;

#define tr(container, it) for ( typeof(container.begin()) it = container.begin(); it!= container.end() ; it++)


// Check TR1 Headers
#if defined(__INTEL_COMPILER) || (defined(__GNUC__) && !defined(__clang__))
  #if defined(__GLIBCXX__)
    #if __GLIBCXX__ >= 20070719 // GCC 4.2.1
      #define HAS_TR1_UNORDERED_MAP
      #define HAS_TR1_UNORDERED_SET
    #endif
  #endif // defined(__GXX_EXPERIMENTAL_CXX0X__) && defined(__GLIBCXX__)

#elif defined(__clang__)
  #if __cplusplus >= 201103L
    #if __has_include(<tr1/unordered_map>)
      #define HAS_TR1_UNORDERED_MAP
    #endif
    #if __has_include(<tr1/unordered_set>)
      #define HAS_TR1_UNORDERED_SET
    #endif
  #endif // __cplusplus >= 201103L
#endif

#if defined(HAS_TR1_UNORDERED_MAP) && defined(HAS_TR1_UNORDERED_SET)
  #define HAS_TR1
#endif


#include <vector>

#ifdef HAS_TR1
  #include <tr1/unordered_map>
  #include <tr1/unordered_set>
  typedef vector<int> veci;
  typedef tr1::unordered_set<int> seti;
  typedef tr1::unordered_map<int, int> mapii ;

  typedef tr1::unordered_map<int, veci> mapi_veci ;
  typedef tr1::unordered_map<int, seti> mapi_seti ;
  typedef tr1::unordered_map<int, mapii> mapi_mapii ;
#else // (slower than HAS_TR1)
  #include <map>
  #include <set>
  typedef vector<int> veci;
  typedef set<int> seti;
  typedef map<int, int> mapii ;

  typedef map<int, veci> mapi_veci ;
  typedef map<int, seti> mapi_seti ;
  typedef map<int, mapii> mapi_mapii ;
#endif




//////////////declarations


class GraphUndir;


bool is_valid_joint_degree(mapi_mapii & nkk);
void neighbor_switch(GraphUndir & G, int & w, seti & unsat, mapii & h_node_residual, int avoid_node_id);
void joint_degree_model(mapi_mapii & nkk, GraphUndir & G);
void load_nkk(char * fname, mapi_mapii & nkk);
void write_graph(char * fname, GraphUndir & G );




