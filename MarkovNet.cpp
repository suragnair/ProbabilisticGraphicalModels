#include <iostream>
#include <cstdio>
#include <cstdlib>
#include <numeric>
#include <vector>
#include <set>
#include <string>
#include <map>
#include <algorithm>
#include <cmath>

#include "Factor.cpp"
#include "FactorGraph.cpp"

#ifndef MARKOVNET_CPP
	#define MARKOVNET_CPP

class MarkovNet
{
	public:
		MarkovNet(int num_nodes, vector<string> node_var_names, vector<int> node_num_vals, vector<vector<int> > adj_list, vector<Factor> factors);
		int num_nodes;
		vector<int> min_fill_ve_order();
		FactorGraph gen_clique_tree(vector<int> elim_ordering);
		FactorGraph gen_bethe_cluster_graph();

		void print(bool print_factors);
		vector<Factor> get_factors();
		
		// sampling based methods
		vector<vector<int> > gibbs_sampler(vector<int> initial_assignments, int burn_in_samples, int check_convergence_every_n_transitions, int check_convergence_versus_last_samples, int max_samples, double epsilon);
		pair<vector<int>, vector<vector<double> > > marginal_prob_dist_from_samples(vector<vector<int> > samples);
		vector<double> marginal_likelihood(vector<int> assignment, vector<vector<double> >& prob_dist);
		
		// parameter learning
		void learn_parameters(vector<vector<int> >& train_data, double learning_rate, double reg_const, double epsilon, int max_iters);		
		
		// assignment 3 specific
		pair<vector<double>, vector<vector<double> > > inference_by_sampling(vector<int> query, vector<int> gnd_assignment);

	private:
		vector<string> node_var_names;
		vector<int> node_num_vals;
		vector<vector<int> > adj_list;
		vector<Factor> factors;
		
		// for Gibbs Sampling
		Factor reduced_factor(string var_to_reduce, vector<int> variable_factors, map<string,int> cur_state_map);
		
		// for parameter learning
		vector<Factor> avg_feature_counts_from_samples(vector<vector<int> >& samples);
		bool update_parameters(vector<Factor>& avg_feature_counts_data, vector<Factor>& avg_feature_counts_param, int num_data_samples, double learning_rate, double reg_const, double epsilon);
};

MarkovNet::MarkovNet(int nn, vector<string> nvn, vector<int> nnv, vector<vector<int> > al, vector<Factor> f)
{
	num_nodes = nn;
	node_var_names = nvn;
	node_num_vals = nnv;
	adj_list = al;
	factors = f;
}

vector<int> MarkovNet::min_fill_ve_order()
{
	vector<bool> marked(num_nodes, false);
	vector<vector<bool> > am(num_nodes, vector<bool>(num_nodes, false));		// adj_matrix
	for (int i = 0 ; i < num_nodes ; i++)
		for (int j = 0 ; j < adj_list[i].size() ; j++)
			am[i][adj_list[i][j]] = true;

	vector<int> order;

	// each iteration
	for (int i = 0 ; i < num_nodes ; i++)
	{
		int min_fill = 99999;		// set to INF
		int min_node = -1;

		for (int j = 0 ; j < num_nodes ; j++)
			if (!marked[j])
			{
				int cur_min_fill = 0;
				vector<int> ngbs;
				for (int k = 0 ; k < num_nodes ; k++)
					if (am[j][k] == true)
						ngbs.push_back(k);

				for (int p = 0 ; p < ngbs.size() ; p++)
				{
					for (int q = p+1 ; q < ngbs.size() ; q++)
					{
						if (not am[ngbs[p]][ngbs[q]]) cur_min_fill++;
						if (cur_min_fill > min_fill) break;
					}
					if (cur_min_fill > min_fill) break;
				}

				if (cur_min_fill < min_fill)
				{
					min_fill = cur_min_fill;
					min_node = j;
				}
			}

		// add fill edges
		vector<int> ngbs;
		for (int k = 0 ; k < num_nodes ; k++)
			if (am[min_node][k] == true)
				ngbs.push_back(k);

		for (int p = 0 ; p < ngbs.size() ; p++)				
			for (int q = p+1 ; q < ngbs.size() ; q++)
				{
					am[ngbs[p]][ngbs[q]] = true;
					am[ngbs[q]][ngbs[p]] = true;
				}

		// remove min_node
		marked[min_node] = true;
		for (int j = 0 ; j < ngbs.size() ; j++)
		{
			am[min_node][ngbs[j]] = false;
			am[ngbs[j]][min_node] = false;
		}
		order.push_back(min_node);	
	}

	return order;
}

vector<Factor> MarkovNet::get_factors() {return factors;}

FactorGraph MarkovNet::gen_clique_tree(vector<int> elim_ordering)
{
	vector<vector<int> > fg_adj_list;
	vector<set<string> > fg_node_scopes;
	vector<vector<int> > fg_node_factors;
	vector<bool> factor_added(factors.size(), false);
	vector<set<string> > tau;	// intermediate factors
	vector<bool> tau_added;		// added to a node

	// elim_ordering.size() == num_nodes
	for (int i = 0 ; i < num_nodes ; i++)
	{
		vector<int> cur_node_factors;
		vector<int> cur_tau_factors;
		set<string> cur_node_scope;

		for (int j = 0 ; j < factors.size() ; j++)
			if (!factor_added[j] and find(factors[j].vars_name.begin(),factors[j].vars_name.end(), node_var_names[elim_ordering[i]]) != factors[j].vars_name.end())
			{
				cur_node_factors.push_back(j);
				factor_added[j] = true;
			}

		for (int j = 0 ; j < tau.size() ; j++)
			if (!tau_added[j] and find(tau[j].begin(), tau[j].end(), node_var_names[elim_ordering[i]]) != tau[j].end())
			{
				cur_tau_factors.push_back(j);
				tau_added[j] = true;
			}

		fg_node_factors.push_back(cur_node_factors);

		// determine scope of current node from factors and taus
		for (int j = 0 ; j < cur_node_factors.size() ; j++)
			set_difference(factors[cur_node_factors[j]].vars_name.begin(), factors[cur_node_factors[j]].vars_name.end(), cur_node_scope.begin(), cur_node_scope.end(), inserter(cur_node_scope, cur_node_scope.end()));
		// ^ added difference of factor_scope - cur_node_scope to cur_node_scope => effectively union

		for (int j = 0 ; j < cur_tau_factors.size() ; j++)
			set_difference(tau[cur_tau_factors[j]].begin(), tau[cur_tau_factors[j]].end(), cur_node_scope.begin(), cur_node_scope.end(), inserter(cur_node_scope, cur_node_scope.end()));

		fg_node_scopes.push_back(cur_node_scope);

		// add edges
		fg_adj_list.push_back(vector<int>());
		for (int j = 0 ; j < cur_tau_factors.size() ; j++)
		{
			fg_adj_list[cur_tau_factors[j]].push_back(i);
			fg_adj_list[i].push_back(cur_tau_factors[j]);
		}

		// add new intermediate tau
		cur_node_scope.erase(node_var_names[elim_ordering[i]]);	// eliminating from scope
		tau.push_back(cur_node_scope);
		tau_added.push_back(false);		
	}

	return FactorGraph(fg_node_scopes.size(), node_var_names, fg_node_scopes, fg_adj_list, factors, fg_node_factors);
}

bool compareFactorScopeSize(Factor a, Factor b) { return (a.vars_name.size() > b.vars_name.size());}

FactorGraph MarkovNet::gen_bethe_cluster_graph()
{
	vector<vector<int> > fg_adj_list;
	vector<set<string> > fg_node_scopes;
	vector<vector<int> > fg_node_factors;
	int fg_num_nodes = 0;	

	// insert variable nodes to Bethe-Cluster graph
	map<string, int> var_name_to_pos;
	for (int i = 0 ; i < node_var_names.size() ; i++)
	{
		var_name_to_pos.insert(pair<string, int>(node_var_names[i],i));
		fg_adj_list.push_back(vector<int>());
		fg_node_factors.push_back(vector<int>());
		fg_node_scopes.push_back(set<string>{node_var_names[i]});		
		fg_num_nodes++ ;
	}

	// sort in decreasing sizes of scope
	sort(factors.begin(), factors.end(), compareFactorScopeSize);

	// absorb smaller factors into bigger ones
	vector<vector<int> > reduced_factors;
	for (int i = 0 ; i < factors.size() ; i++)
	{
		bool absorbed = false;
		for (int j = 0 ; j < reduced_factors.size() ; j++)
			if (includes(factors[reduced_factors[j][0]].vars_name.begin(), factors[reduced_factors[j][0]].vars_name.end(), factors[i].vars_name.begin(), factors[i].vars_name.end()))
			{
				absorbed = true;
				reduced_factors[j].push_back(i);
				break;
			}

		if (not absorbed) reduced_factors.push_back(vector<int>{i});
	}

	// add factor nodes to Bethe-Cluster graph
	for (int i = 0 ; i < reduced_factors.size() ; i++)
	{
		fg_adj_list.push_back(vector<int>());
		fg_node_scopes.push_back(set<string>(factors[reduced_factors[i][0]].vars_name.begin(),factors[reduced_factors[i][0]].vars_name.end()));
		fg_node_factors.push_back(reduced_factors[i]);

		for (int k = 0 ; k < factors[reduced_factors[i][0]].vars_name.size() ; k++)
		{
			fg_adj_list[var_name_to_pos[factors[reduced_factors[i][0]].vars_name[k]]].push_back(fg_num_nodes);
			fg_adj_list[fg_num_nodes].push_back(var_name_to_pos[factors[reduced_factors[i][0]].vars_name[k]]);
		}

		fg_num_nodes++;
	}

	return FactorGraph(fg_num_nodes, node_var_names, fg_node_scopes, fg_adj_list, factors, fg_node_factors);
}	

pair<vector<int>, vector<vector<double> > > MarkovNet::marginal_prob_dist_from_samples(vector<vector<int> > samples)
{
	// returns max marginal assignment for each node, and also a distribution over all its values
	vector<vector<double> > prob_dist(num_nodes);
	for (int i = 0 ; i < num_nodes ; i++)
		prob_dist[i] = vector<double>(node_num_vals[i], 1.0/(samples.size()+node_num_vals[i]));	// smoothing!
	
	for (int i = 0 ; i < samples.size() ; i++)
		for (int j = 0 ; j < num_nodes ; j++)
			prob_dist[j][samples[i][j]] += 1.0/(samples.size()+node_num_vals[j]);
	
	vector<int> max_marginal_assignment(num_nodes);
	
	for (int i = 0 ; i < num_nodes ; i++)
	{
		vector<double>::iterator max_it = max_element(prob_dist[i].begin(), prob_dist[i].end());
		max_marginal_assignment[i] = distance(prob_dist[i].begin(), max_it);
	}

	return pair<vector<int>, vector<vector<double> > >(max_marginal_assignment, prob_dist);	
}

vector<double> MarkovNet::marginal_likelihood(vector<int> assignment, vector<vector<double> >& prob_dist)
{
	vector<double> loglikelihood;
	
	for (int i = 0 ; i < num_nodes ; i++)
		loglikelihood.push_back(log(prob_dist[i][assignment[i]]));

	return loglikelihood;
}

vector<vector<int> > MarkovNet::gibbs_sampler(vector<int> initial_assignments, int burn_in_samples, int check_convergence_every_n_transitions, int check_convergence_versus_last_samples, int max_samples, double epsilon)
{
	// initial_assignment to a variable is -1 if no assignment (unobserved), else index of assignment (observed)
	// returns samples after burn_in_samples till convergence
	
	vector<vector<int> > samples;
	vector<int> cur_state(num_nodes, 0);
	map<string, int> cur_state_map;
	
	// initialise state
	for (int i = 0 ; i < num_nodes ; i++)
	{
		if (initial_assignments[i] != -1)		
			cur_state[i] = initial_assignments[i];			
		
		cur_state_map.insert(pair<string,int>(node_var_names[i], cur_state[i]));
	}	
	
	// initialise variable factors
	vector<vector<int> > variable_factors_indices(num_nodes, vector<int>());
	for (int i = 0 ; i < num_nodes ; i++)
		for (int j = 0 ; j < factors.size() ; j++)
			if (find(factors[j].vars_name.begin(),factors[j].vars_name.end(), node_var_names[i]) != factors[j].vars_name.end())
				variable_factors_indices[i].push_back(j); 

	bool converged = false;
	int num_samples = 0;
	int num_n_step_transitions = 0;
		
	// start sampling
	while (not converged and num_samples - burn_in_samples < max_samples)
	{		
		for (int i = 0 ; i < num_nodes ; i++)
			if (initial_assignments[i] == -1)
			{
				// sampling ith variable given others, if ith variable is not observed				
				int var_sample = reduced_factor(node_var_names[i], variable_factors_indices[i], cur_state_map).var_sample(node_var_names[i], cur_state_map);
								
				// updates
				cur_state[i] = var_sample;
				cur_state_map[node_var_names[i]] = var_sample;												
				num_samples ++;
				if (num_samples > burn_in_samples) 
				{
					samples.push_back(cur_state);
				}
			}
		
		num_n_step_transitions ++; 
			
		if ((num_samples > burn_in_samples) and (num_n_step_transitions%check_convergence_every_n_transitions==0) and (num_samples - burn_in_samples > 2 * check_convergence_versus_last_samples))
		{
			converged = true;
			// TODO: check if the last check_convergence_versus_last samples give the same expected value wise distribution as all the samples till samples.size() - last check_convergence_versus_last
			for (int i = 0 ; i < num_nodes - 1 ; i++)
			{
				double first_sample_avg = 0.0;
				double all_sample_avg = 0.0;
				
				for (int j = 0 ; j < samples.size() - check_convergence_versus_last_samples ; j++)
					first_sample_avg += samples[j][i];
				
				all_sample_avg = first_sample_avg;
				for (int j = samples.size() - check_convergence_versus_last_samples ; j < samples.size() ; j++)
					all_sample_avg += samples[j][i];
				
				first_sample_avg /= samples.size() - check_convergence_versus_last_samples;
				all_sample_avg /= samples.size();
				
				if (abs(first_sample_avg - all_sample_avg) > epsilon)
				{
					converged = false;
					break;
				}
			}
		}
	}	
	
	if (converged) cout << "Gibbs Sampling converged after        : " << num_samples << " samples\n";
	else cout << "Gibbs Sampling did not converge after : " << num_samples << " samples\n";
	return samples;	
}

Factor MarkovNet::reduced_factor(string var_to_reduce, vector<int> variable_factors_indices, map<string,int> cur_state_map)
{
	Factor reduced;
	
	for (int i = 0 ; i < variable_factors_indices.size() ; i++)
	{
		int var_to_red_ind_in_cur_factor;
		vector<int> assignment(factors[variable_factors_indices[i]].num_vars, -1);
		for (int j = 0 ; j < factors[variable_factors_indices[i]].num_vars ; j++)
		{
			if (factors[variable_factors_indices[i]].vars_name[j] == var_to_reduce)
				var_to_red_ind_in_cur_factor = j;
			assignment[j] = cur_state_map[factors[variable_factors_indices[i]].vars_name[j]];
		}
		
		vector<double> pots;
		for (int j = 0 ; j < factors[variable_factors_indices[i]].num_vals_vars[var_to_red_ind_in_cur_factor] ; j++)
		{
			assignment[var_to_red_ind_in_cur_factor] = j;
			pots.push_back(factors[variable_factors_indices[i]].pot_at(assignment));
		}
		
		reduced = reduced * Factor(1, vector<string>(1,var_to_reduce), vector<int>(1, factors[variable_factors_indices[i]].num_vals_vars[var_to_red_ind_in_cur_factor]), pots);
	}
	
	return reduced;
}

pair<vector<double>, vector<vector<double> > > MarkovNet::inference_by_sampling(vector<int> query, vector<int> gnd_assignment)
{
	// query consists of -1 if variable not observed, else value of observation
	// gnd_assignment has the ground assingment
	// return vector<double> is of the form {total_correct_vars, num_missing_vars, LL_gnd_assgnmnt}
	// also returns the probability distribution over each variable
	// prediction and LL at marginal probability level
	
	vector<vector<int> > samples = gibbs_sampler(query, 5000, 100, 4000, 50000, 0.01);
	pair<vector<int>, vector<vector<double> > > dist = marginal_prob_dist_from_samples(samples);
	vector<double> gnd_ll_vars = marginal_likelihood(gnd_assignment, dist.second);
	double gnd_ll = accumulate(gnd_ll_vars.begin(), gnd_ll_vars.end(), 0.0);		// note that LL will be 0 for observed vars (because of how sampling is done)
	double correct_vars_pred = 0;
	double missing_vars_total = 0;
	
	for (int i = 0 ; i < query.size() ; i++)
		if (query[i]==-1)
		{
			missing_vars_total ++;
			if (gnd_assignment[i] == dist.first[i]) correct_vars_pred++;
		}
		
	return pair<vector<double>, vector<vector<double> > >(vector<double>{missing_vars_total, correct_vars_pred, gnd_ll}, dist.second);
}


void MarkovNet::learn_parameters(vector<vector<int> >& train_data, double learning_rate, double reg_const, double epsilon, int max_iters)
{
	// assumes structure and factors have been initialised
	
	// each factor phi in a factor table is the exp(lambda) in the equivalent log linear model
	// to learn the phis, we update the lambdas first using the update equation
	// dLL/dl_k = m * [ (avg value of f_k from data) - (avg value of f_k from parameters) ] - 2*C*l_k {l_k is lambda corresponding to feature f_k}
	// refer class notes for more details
	
	// calculate expected feature counts from data, store as factor tables for each feature, ie each row of factor table (but they don't act as factors!)
	vector<Factor> avg_feature_counts_data = avg_feature_counts_from_samples(train_data);

	bool converged = false;
	vector<Factor> avg_feature_counts_param;
	vector<vector<int> > samples; 
	int iters = 0;
	
	while (not converged and iters < max_iters)
	{
		samples = gibbs_sampler(vector<int>(num_nodes, -1), 3000, 100, 1000, 20000, 0.001);
		avg_feature_counts_param = avg_feature_counts_from_samples(samples); 
		
		converged = update_parameters(avg_feature_counts_data, avg_feature_counts_param, train_data.size(), learning_rate, reg_const, epsilon);
		factors[0].print();
		iters++;
		
		cout << "Num iterations : " << iters << endl; 
	}
}

vector<Factor> MarkovNet::avg_feature_counts_from_samples(vector<vector<int> >& samples)
{
	vector<Factor> avg_feat_counts;
	map<string, int> var_name_to_pos;
	for (int i = 0 ; i < node_var_names.size() ; i++)
		var_name_to_pos.insert(pair<string, int>(node_var_names[i],i));
	
	for (int i = 0 ; i < factors.size() ; i++)
	{
		// build a table corresponding to this factor table that stores the average counts of each assignment in train_data
		avg_feat_counts.push_back(Factor(factors[i].num_vars, factors[i].vars_name, factors[i].num_vals_vars, vector<double>(factors[i].potentials.size(), 0.0)));
		
		vector<int> assignment_map;
		for (int j = 0 ; j < factors[i].num_vars ; j++)
			assignment_map.push_back(var_name_to_pos[factors[i].vars_name[j]]);
		
		for (int j = 0 ; j < samples.size() ; j++)
		{
			vector<int> assignment;
			for (int k = 0 ; k < assignment_map.size() ; k++)
				assignment.push_back(samples[j][assignment_map[k]]);
			
			avg_feat_counts[i].potentials[factors[i].flat_index_from_assignment(assignment)] += 1.0/samples.size();
		}
	}
	
	return avg_feat_counts;
}

bool MarkovNet::update_parameters(vector<Factor>& avg_feature_counts_data, vector<Factor>& avg_feature_counts_param, int num_data_samples, double learning_rate, double reg_const, double epsilon)
{
	bool converged = true;
	int cur_num_features;
	double cur_lambda;
	double new_lambda;
	double delta_lambda;
	double cur_reg;
	for (int i = 0 ; i < factors.size() ; i++)
	{
		cur_num_features = accumulate(factors[i].num_vals_vars.begin(), factors[i].num_vals_vars.end(), 1, multiplies<int>());
		for (int j = 0 ; j < cur_num_features ; j++)
		{
			cur_lambda = log(factors[i].potentials[j]);
			delta_lambda = avg_feature_counts_data[i].potentials[j] - avg_feature_counts_param[i].potentials[j] ;
			cur_reg = - (2 * reg_const * cur_lambda)/num_data_samples;		// L2 regularization
			
			new_lambda = cur_lambda + learning_rate * (delta_lambda + cur_reg);						
			
			factors[i].potentials[j] = exp(new_lambda);			
			if (abs(exp(new_lambda) - exp(cur_lambda)) > epsilon) converged = false;			
		}
	}
	
	return converged;
}

void MarkovNet::print(bool print_factors)
{
	cout << "------- MARKOV NET -------" << endl << endl;
	cout << "Num Nodes : " << num_nodes << endl << endl;

	cout << "Node Names\n==========\n";
	for (int i = 0 ; i < num_nodes ; i++)
		cout << i << " : " << node_var_names[i] << endl;

	cout << "\nFactor Scopes\n=============\n";
	for (int i = 0 ; i < factors.size() ; i++)
	{
		cout << i << " : " ;
		for (int j = 0 ; j < factors[i].vars_name.size() ; j++)
			cout << factors[i].vars_name[j] << " ";
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

	if (print_factors)
	{
		cout << "\nFactors\n=======\n";
		for (int i = 0 ; i < factors.size() ; i++)
			factors[i].print();
	}

	cout << "----------------------------" << endl << endl;
}

#endif
