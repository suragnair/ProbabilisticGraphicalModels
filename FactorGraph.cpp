using namespace std;
#include <math.h>

#ifndef FACTORGRAPH_CPP
	#define FACTORGRAPH_CPP

// TODO: test base case (one clique node, 2 nodes)

class FactorGraph
{
	public:
		FactorGraph(int num_nodes, vector<string> var_names, vector<set<string> > node_scopes, vector<vector<int> > adj_list, vector<Factor> factors, vector<vector<int> > node_factors);
		FactorGraph();
		void MessagePassing(int root, Factor (Factor::*margin_op)(string));	// populates node_marginals
		void BeliefProp(double epsilon, int max_iter, Factor (Factor::*margin_op)(string), bool silent);	// populates node_marginals
		pair<vector<int>, vector<double> > max_marginal_assignment(Factor (Factor::*margin_op)(string));	// after node_marginals is populated		
		vector<double> marginal_likelihood(bool run_bp, vector<int> assignments);

		void print(bool marginals);

		int num_nodes;
		vector<Factor> node_marginals;
		
	//private:
		vector<set<string> > node_scopes;
		vector<string> var_names;
		vector<vector<int> > adj_list;
		vector<Factor> factors;
		vector<vector<int> > node_factors;		

		// methods for MessagePassing
		void up_pass(int start_node, int parent, vector<map<int,Factor> >& messages, vector<bool>& visited, Factor (Factor::*margin_op)(string));
		void down_pass(int start_node, int parent, vector<map<int,Factor> >& messages, Factor (Factor::*margin_op)(string));
		
};

FactorGraph::FactorGraph(int nn, vector<string> vn, vector<set<string> > ns, vector<vector<int> > al, vector<Factor> f, vector<vector<int> > nf)
{
	num_nodes = nn;
	var_names = vn;
	node_scopes = ns;
	adj_list = al;
	factors = f;
	node_factors = nf;
	node_marginals = vector<Factor>(num_nodes);
}

FactorGraph::FactorGraph() {}	// empty constructor

void FactorGraph::MessagePassing(int root, Factor (Factor::*margin_op)(string))
{
	// works only if cluster graph is clique tree
	vector<map<int,Factor> > messages(num_nodes, map<int,Factor>());		// ith element stores messages coming to i, ie delta_(j->i)

	// node_marginals (beliefs) are set in down_pass
	// loop over in case of multiple connected components (after calling from root)
	vector<bool> visited(num_nodes, false);

	up_pass(root, -1, messages, visited, margin_op);
	down_pass(root, -1, messages, margin_op);	

	for (int i = 0 ; i < num_nodes ; i++)
		if (not visited[i])
		{
			up_pass(i, -1, messages, visited, margin_op);
			down_pass(i, -1, messages, margin_op);
		}
}

void FactorGraph::up_pass(int start_node, int parent, vector<map<int,Factor> >& messages, vector<bool>& visited, Factor (Factor::*margin_op)(string))
{	
	visited[start_node] = true;
	vector<int> down_ngbs = adj_list[start_node];
	down_ngbs.erase(remove(down_ngbs.begin(), down_ngbs.end(), parent), down_ngbs.end());
	
	Factor up_message = Factor();
	for (int j = 0 ; j < node_factors[start_node].size() ; j++)
		up_message = up_message * factors[node_factors[start_node][j]];

	for (int i = 0 ; i < down_ngbs.size() ; i++)	// down_ngbs.size() == 0 for leaf node
		up_pass(down_ngbs[i], start_node, messages, visited, margin_op);

	map<int, Factor>::iterator it;
	for (it = messages[start_node].begin() ; it != messages[start_node].end() ; it++)
		up_message = up_message * it->second;

	if (parent != -1)	// not root node => sum up
	{
		// sum out
		set<string> scope_diff;
		set_difference(node_scopes[start_node].begin(), node_scopes[start_node].end(), node_scopes[parent].begin(), node_scopes[parent].end(), inserter(scope_diff, scope_diff.end()));

		set<string>::iterator it;
		for (it = scope_diff.begin() ; it != scope_diff.end() ; it++)
			up_message = (up_message.*margin_op)(*it);

		messages[parent].insert(pair<int, Factor>(start_node, up_message));
	}
}

void FactorGraph::down_pass(int start_node, int parent, vector<map<int,Factor> >& messages, Factor (Factor::*margin_op)(string))
{
	vector<int> down_ngbs = adj_list[start_node];
	down_ngbs.erase(remove(down_ngbs.begin(), down_ngbs.end(), parent), down_ngbs.end());

	Factor belief = Factor();
	for (int i = 0 ; i < node_factors[start_node].size() ; i++)
		belief = belief * factors[node_factors[start_node][i]];

	map<int, Factor>::iterator it;
	for (it = messages[start_node].begin() ; it != messages[start_node].end() ; it++)
		belief = belief * it->second;	

	// setting node marginal
	node_marginals[start_node] = belief;

	for (int i = 0 ; i < down_ngbs.size() ; i++)
	{
		Factor down_message = belief;
		down_message = down_message/messages[start_node][down_ngbs[i]];				

		// sum out
		set<string> scope_diff;
		set_difference(node_scopes[start_node].begin(), node_scopes[start_node].end(), node_scopes[down_ngbs[i]].begin(), node_scopes[down_ngbs[i]].end(), inserter(scope_diff, scope_diff.end()));
		set<string>::iterator it;
		for (it = scope_diff.begin() ; it != scope_diff.end() ; it++)
			down_message = (down_message.*margin_op)(*it);

		messages[down_ngbs[i]].insert(pair<int, Factor>(start_node, down_message));

		down_pass(down_ngbs[i], start_node, messages, margin_op);
	}
}

void FactorGraph::BeliefProp(double epsilon, int max_iter, Factor (Factor::*margin_op)(string), bool silent)
{	
	// initialise
	vector<map<int,Factor> > cur_messages(num_nodes, map<int,Factor>());	// ith element stores messages coming to i, ie delta_(j->i)
	vector<map<int,Factor> > new_messages(num_nodes, map<int,Factor>());

	for (int i = 0 ; i < num_nodes ; i++)
		for (int j = 0 ; j < adj_list[i].size() ; i ++)
		{
			// delta_(i->j)
			cur_messages[j].insert(pair<int, Factor>(i, Factor()));
			new_messages[j].insert(pair<int, Factor>(i, Factor()));
		}

	vector<Factor> cur_beliefs(num_nodes, Factor());
	vector<Factor> new_beliefs(num_nodes, Factor());

	for (int i = 0 ; i < num_nodes ; i++)
		for (int j = 0 ; j < node_factors[i].size() ; j++)
			cur_beliefs[i] = cur_beliefs[i] * factors[node_factors[i][j]];

	bool converged = false;
	int iter = 0 ;

	while (!converged and iter < max_iter)
	{	
		converged = true;
		// cout << "===============\nITERATION " << iter << "\n===============\n";

		// update messages
		for (int i = 0 ; i < num_nodes ; i++)
			for (int j = 0 ; j < adj_list[i].size() ; j++)
			{					
				// delta_(i->adj_list[i][j])
				// cout << "delta_(" << i << "->" << adj_list[i][j] << ")"<<endl;				
				Factor new_message = cur_beliefs[i]/cur_messages[i][adj_list[i][j]];
				
				// sum out
				set<string> scope_diff;
				set_difference(node_scopes[i].begin(), node_scopes[i].end(), node_scopes[adj_list[i][j]].begin(), node_scopes[adj_list[i][j]].end(), inserter(scope_diff, scope_diff.end()));
				set<string>::iterator it;
								
				for (it = scope_diff.begin() ; it != scope_diff.end() ; it++)					
					new_message = (new_message.*margin_op)(*it);		
				
				new_messages[adj_list[i][j]][i] = new_message;									
				new_messages[adj_list[i][j]][i].normalize();
				//cout << "delta_(" << i << "->" << adj_list[i][j] << ")\n";
				// new_messages[adj_list[i][j]][i].print();
			}

		// update beliefs
		for (int i = 0 ; i < num_nodes ; i++)
		{
			Factor new_belief = Factor();
			for (int j = 0 ; j < node_factors[i].size() ; j++)
				new_belief = new_belief * factors[node_factors[i][j]];

			map<int, Factor>::iterator it;
			for (it = new_messages[i].begin() ; it != new_messages[i].end() ; it++)
				new_belief = new_belief * it->second;	

			new_beliefs[i] = new_belief;
			//cout << "belief_" << i << "\n";
			//new_belief.print();

			for (int j = 0 ; j < new_beliefs[i].potentials.size() ; j++)
				if (abs(cur_beliefs[i].potentials[j] - new_beliefs[i].potentials[j]) > epsilon)
					converged = false;
		}

		cur_beliefs = new_beliefs;
		cur_messages = new_messages;
		iter++ ;
	}

	for (int i = 0 ; i<num_nodes ; i++)
		node_marginals[i] = cur_beliefs[i];

	if (converged and !silent) cout << "BeliefProp converged in " << iter << " iterations\n\n";
	else if (!silent) cout << "BeliefProp did not converge in " << max_iter << " iterations\n\n";

}

pair<vector<int>, vector<double> > FactorGraph::max_marginal_assignment(Factor (Factor::*margin_op)(string))
{
	// can speed up
	vector<int> assignments(var_names.size(), -1);
	vector<double> loglikelihood; 	// as defined in problem statement for max_marginal, makes sense only for margin_op == sum_out
	for (int i = 0 ; i < assignments.size() ; i++)
	{
		int j ;
		for (j = 0 ; j < node_marginals.size() ; j++)
			if (find(node_marginals[j].vars_name.begin(), node_marginals[j].vars_name.end(), var_names[i]) != node_marginals[j].vars_name.end())
				break;

		Factor var_marginal = node_marginals[j];
		var_marginal.normalize();
		for (int k = 0 ; k < node_marginals[j].vars_name.size() ; k++)
			if (node_marginals[j].vars_name[k] != var_names[i])
				var_marginal = (var_marginal.*margin_op)(node_marginals[j].vars_name[k]);
		
		vector<double>::iterator max_it = max_element(var_marginal.potentials.begin(), var_marginal.potentials.end());
		loglikelihood.push_back(log(*max_it));
		assignments[i] = distance(var_marginal.potentials.begin(), max_it);
	}

	return pair<vector<int>, vector<double> >(assignments, loglikelihood);
}

vector<double> FactorGraph::marginal_likelihood(bool run_bp, vector<int> assignments)
{
	// run quick BP (for max) to find the marginal at each node (not MP since it could be Bethe Cluster)
	if (run_bp) this->BeliefProp(0.01, 1000, &Factor::sum_out, true);
	vector<double> loglikelihood; 

	for (int i = 0 ; i < assignments.size() ; i++)
	{
		int j ;
		for (j = 0 ; j < node_marginals.size() ; j++)
			if (find(node_marginals[j].vars_name.begin(), node_marginals[j].vars_name.end(), var_names[i]) != node_marginals[j].vars_name.end())
				break;
		
		Factor var_marginal = node_marginals[j];
		var_marginal.normalize();
		for (int k = 0 ; k < node_marginals[j].vars_name.size() ; k++)
			if (node_marginals[j].vars_name[k] != var_names[i])
				var_marginal = var_marginal.sum_out(node_marginals[j].vars_name[k]);
				
		loglikelihood.push_back(log(var_marginal.potentials[assignments[i]]));		
	}

	return loglikelihood;
}

void FactorGraph::print(bool marginals)
{	
	cout << "------- FACTOR GRAPH -------" << endl << endl;
	cout << "Num Nodes : " << num_nodes << endl << endl;

	cout << "Node Scopes\n===========\n";
	for (int i = 0 ; i < node_scopes.size() ; i++)
	{
		set<string>::iterator it;
		cout << i << " : " ;
		for (it = node_scopes[i].begin() ; it != node_scopes[i].end(); it++)
			cout << *it << " ";
		cout << endl;
	}

	cout << "\nFactor Scopes\n=============\n";
	for (int i = 0 ; i < factors.size() ; i++)
	{
		cout << i << " : " ;
		for (int j = 0 ; j < factors[i].vars_name.size() ; j++)
			cout << factors[i].vars_name[j] << " ";
		cout << endl;
	}

	cout << "\nNode Factors\n============\n";
	for (int i = 0 ; i < node_factors.size() ; i++)
	{
		vector<int>::iterator it;
		cout << i << " : " ;
		for (it = node_factors[i].begin() ; it != node_factors[i].end(); it++)
			cout << *it << " ";
		cout << endl;
	}

	cout << "\nAdjaceny List\n=============\n";
	for (int i = 0 ; i < adj_list.size() ; i++)
	{
		vector<int>::iterator it;
		cout << i << " : " ;
		for (it = adj_list[i].begin() ; it != adj_list[i].end(); it++)
			cout << *it << " ";
		cout << endl;
	}

	if (marginals)
	{
		cout << "\nMarginals\n=========\n";
		for (int i = 0 ; i < num_nodes ; i++)
		{
			node_marginals[i].normalize();
			node_marginals[i].print();
		}
	}
	cout << "----------------------------" << endl << endl;
}

#endif