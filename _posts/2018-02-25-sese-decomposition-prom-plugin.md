---
title: "SESE-based Decomposition ProM Plugin"
date: 2018-02-25
type: posts
published: false
comments: true
categories: [ "prom", "process mining" ]
---

#### TL;DR
New feature in the AcceptingPetriNetDecomposer plugin to allow SESE-based decomposition.


#### How?
This decomposition strategy tries to break down a petri net into fragments with a single entry and a single exit transition. The technique involves first treating a petri net a directed graph and both places and transitions as the same kind of nodes. Then, a Refined Process Structure Tree (RPST) is built where the root is the entire graph and each child node is a fragment with a single entry and single exit. The leaf nodes are single edges which are canonical fragments since they connect two nodes, i.e., has a single entry and a single exit. Since a valid decomposition is required which restricts to decomposing on unique visible transitions, a so-called bridging technique is used to modify fragments have places as entries and/or exits. To bridge a particular fragment, one finds other fragments that share the entry or exit place and add the preceding or subsequent transitions to the fragment so that the modified fragment is splitting on transitions rather than places. Of course, there are several cases that one has to take care of, e.g., non-unique/visible transitions. Interested readers are referred to the paper - [Single-Entry-Single-Exit Decomposed Conformance Checking](http://www.jorgemunozgama.com/data/uploads/pub/papers/is14.pdf) for the more juicy detail.

#### What's so great?
- Single-entry-single-exit decomposition is intuitively much more inline with how people think about how processes should be broken. The user study in the paper - [Hierarchical Conformance Checking of Process Models Based on Event Logs](http://www.jorgemunozgama.com/data/uploads/pub/papers/petrinets13.pdf) shows that human users tend break down process models as fragments with single entries and exits.
- It leads to much more significant performance gains in computation time as opposed to maximal and passage-based decompositions for decomposed conformance checking.

#### Where is it?
- Can be accessed through the ProMNightly build.
- [SVN](https://svn.win.tue.nl/trac/prom/browser/Packages/AcceptingPetriNetDecomposer)

#### Instructions on using it:
1. Import an accepting petri net into ProM.  
![Screenshot of **Create Matrix** plugin](/assets/images/2018/2018-02-25-sese-decomposition-prom-plugin/createMatrixPlugin.jpg)  
2. Create causal matrix (**Create Matrix**) using the accepting petri net.  
![Screenshot of **Create Graph** plugin](/assets/images/2018/2018-02-25-sese-decomposition-prom-plugin/createGraphPlugin.jpg)
3. Create causal graph (**Create Graph**) using the causal matrix.  
![Screenshot of **Create Clusters** plugin](/assets/images/2018/2018-02-25-sese-decomposition-prom-plugin/createClustersPlugin.jpg)
4. Create activity clusters (**Create Clusters**) using the causal graph.  
![Screenshot of **Split Accepting Petri Net** plugin configurations](/assets/images/2018/2018-02-25-sese-decomposition-prom-plugin/splitApnPlugin.jpg)  
![Screenshot of max arcs per subnet configuration](/assets/images/2018/2018-02-25-sese-decomposition-prom-plugin/chooseMaxArc.jpg)  
5. Start accepting petri net decomposer plugin (**Split Accepting Petri Net**) using the accepting petri net and activity clusters. Choose the SESE-based decomposition strategy and adjust the maximum number of arcs per subnet if necessary.   
