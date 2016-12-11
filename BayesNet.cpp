#include <iostream>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <numeric>
#include <vector>
#include <set>
#include <string>
#include <map>
#include <algorithm>
#include <cmath>

#include "Factor.cpp"
#include "MarkovNet.cpp"

using namespace std;

class BayesNet
{
	public:
		BayesNet(int num_nodes, vector<string> node_var_names, vector<vector<string> > node_vals, vector<vector<int> > adj_list, vector<Factor> CPT);
		BayesNet(string bif_file);
		int num_nodes;
		vector<string> node_var_names; 
		map<string,int> node_name_to_index;
		vector<map<string,int> > node_val_to_index;
		vector<vector<string> > node_vals;
		vector<vector<int> > adj_list;
		vector<vector<int> > rev_adj_list;
		vector<Factor> CPT;	
		
		vector<vector<int> > read_data_file(string filename);
		void inference_on_file(string test_filename, string gnd_filename, string out_filename, MarkovNet mn);
		void write_params_to_file(string filename, bool write_as_markov_net);
		void print();
		
		MarkovNet moralize_bn();
		
		// parameter learning 
		void learn_parameters(vector<vector<int> > train_data);
		
	private:
		void init_bn(int num_nodes, vector<string> node_var_names, vector<vector<string> > node_vals, vector<vector<int> > adj_list, vector<Factor> CPT);
		
		// for parameter learning
		void normalize_CPT_from_counts();
		
};

BayesNet::BayesNet(int nn, vector<string> nvn, vector<vector<string> > nv, vector<vector<int> > al, vector<Factor> cpt)
{
	// CPTs are stored as Factors such that last variable is the child of the remaining variables in the BayesNet
	init_bn(nn, nvn, nv, al, cpt);
}

void BayesNet::init_bn(int nn, vector<string> nvn, vector<vector<string> > nv, vector<vector<int> > al, vector<Factor> cpt)
{
	num_nodes = nn;
	node_var_names = nvn;	
	node_vals = nv;
	adj_list = al;
	CPT = cpt;
	rev_adj_list = vector<vector<int> >(nn, vector<int>());
	
	for (int i = 0 ; i < node_var_names.size() ; i++)
		node_name_to_index.insert(pair<string,int>(node_var_names[i],i));
		
	for (int i = 0 ; i < node_vals.size() ; i++)
	{
		node_val_to_index.push_back(map<string,int>());
		for (int j = 0 ; j < node_vals[i].size() ; j++)
			node_val_to_index[i].insert(pair<string,int>(node_vals[i][j],j));
	}
	
	for (int i = 0 ; i < num_nodes ; i++)
		for (int j = 0 ; j < adj_list[i].size() ; j++)
			rev_adj_list[adj_list[i][j]].push_back(i);
}

void BayesNet::learn_parameters(vector<vector<int> > train_data)
{
	// assumes all CPT values are set to 1.0 initially
	for (int i = 0 ; i < CPT.size() ; i++)
	{
		vector<int> assignment_map;
		for (int j = 0 ; j < CPT[i].num_vars ; j++)
			assignment_map.push_back(node_name_to_index[CPT[i].vars_name[j]]);
		
		for (int j = 0 ; j < train_data.size() ; j++)
		{
			vector<int> assignment;
			for (int k = 0 ; k < assignment_map.size() ; k++)
				assignment.push_back(train_data[j][assignment_map[k]]);
			
			CPT[i].potentials[CPT[i].flat_index_from_assignment(assignment)] += 1.0;
		}		
	}

	normalize_CPT_from_counts();
}

void BayesNet::normalize_CPT_from_counts()
{
	// each CPT is distribution of last variable in CPT given remaining variables
	// to normalize each CPT after counting from data
	// within each CPT, normalize for each assignment to remaining variables
	
	for (int i = 0 ; i < CPT.size() ; i++)
	{
		vector<int> assignment_to_remaining(CPT[i].num_vars-1, 0);
		int tot_assn_to_rem = accumulate(CPT[i].num_vals_vars.begin(), CPT[i].num_vals_vars.end()-1, 1, multiplies<int>()); // total assignments to the remaining variables
		vector<int> full_assignment;
		
		for (int j = 0 ; j < tot_assn_to_rem ; j++)
		{
			int denominator = 0;
			for (int k = 0 ; k < CPT[i].num_vals_vars[CPT[i].num_vars-1] ; k++)	// cycle through all values
			{
				full_assignment = assignment_to_remaining;
				full_assignment.push_back(k);
				denominator += CPT[i].pot_at(full_assignment);
			}
			
			// normalize
			for (int k = 0 ; k < CPT[i].num_vals_vars[CPT[i].num_vars-1] ; k++)
			{
				full_assignment = assignment_to_remaining;
				full_assignment.push_back(k);
				CPT[i].potentials[CPT[i].flat_index_from_assignment(full_assignment)] /= denominator;
			}
			
			CPT[i].increment(assignment_to_remaining, vector<int>(CPT[i].num_vals_vars.begin(), CPT[i].num_vals_vars.end()-1));
		}		
	}
}

MarkovNet BayesNet::moralize_bn()
{
	vector<vector<int> > mn_adj_list = adj_list;
	
	// making edges undirected
	for (int i = 0 ; i < num_nodes ; i++)
		for (int j = 0 ; j < rev_adj_list[i].size() ; j++)
			mn_adj_list[i].push_back(rev_adj_list[i][j]);
	
	// add edges between parents
	for (int i = 0 ; i < num_nodes ; i++)
		for (int j = 0 ; j < rev_adj_list[i].size() ; j++)
			for (int k = j + 1 ; k < rev_adj_list[i].size() ; k++)
				if (find(mn_adj_list[rev_adj_list[i][j]].begin(), mn_adj_list[rev_adj_list[i][j]].end(), rev_adj_list[i][k]) == mn_adj_list[rev_adj_list[i][j]].end())
				{
					mn_adj_list[rev_adj_list[i][j]].push_back(rev_adj_list[i][k]);
					mn_adj_list[rev_adj_list[i][k]].push_back(rev_adj_list[i][j]);
				}
	
	vector<int> mn_node_num_vals;	
	for (int i = 0 ; i < num_nodes ; i++)
		mn_node_num_vals.push_back(node_vals[i].size());
	
	return MarkovNet(num_nodes, node_var_names, mn_node_num_vals, mn_adj_list, CPT);	
}

BayesNet::BayesNet(string bif_file)
{
	ifstream infile(bif_file);
	string line;
	int num_nodes = 0;
	vector<string> node_var_names;
	map<string, int> node_name_to_index;
	vector<vector<string> > node_vals;
	vector<vector<int> > adj_list;
	vector<Factor> cpt;
	
	if (infile.is_open())
	{
		string temp = "";
			
		while (true)
		{
			getline(infile,line);
			stringstream ss(line);
			ss >> temp;
			if (temp=="probability") break;
			else if (temp!="variable") continue;
			else	// variable
			{
				ss >> temp;
				node_var_names.push_back(temp);
				node_name_to_index.insert(pair<string,int>(temp,num_nodes));
				node_vals.push_back(vector<string>());
				num_nodes ++;
				getline(infile, line);
				ss = stringstream(line);
				while(temp!="{") ss >> temp;
				while(true)
				{						
					ss >> temp;
					if (temp=="};") break;
					if (temp[temp.size()-1]==',') node_vals[num_nodes-1].push_back(temp.substr(0,temp.size()-1));
					else node_vals[num_nodes-1].push_back(temp);
				}	
			}				
		}
		
		adj_list = vector<vector<int> >(num_nodes, vector<int>());
		
		while (true) 
		{		
			string prob_of;	
			stringstream ss(line);
			ss >> temp;
			if (temp=="probability")
			{
				vector<string> cur_factor_vars_name;
				ss >> temp; ss >> temp;
				prob_of = temp;

				while (true)
				{
					ss >> temp;
					if (temp==")") break;
					if (temp=="|") continue;
					if (temp[temp.size()-1]==',') cur_factor_vars_name.push_back(temp.substr(0,temp.size()-1));
					else cur_factor_vars_name.push_back(temp);
				}
				cur_factor_vars_name.push_back(prob_of); 	// last variable
				
				vector<int> cur_factor_vars_id;
				for (int i = 0 ; i < cur_factor_vars_name.size() ; i++)
					cur_factor_vars_id.push_back(node_name_to_index[cur_factor_vars_name[i]]);
				vector<int> cur_factor_num_vals_vars;
				int tot_pots = 1;
				for (int i = 0 ; i < cur_factor_vars_id.size() ; i++)
				{
					tot_pots *= node_vals[cur_factor_vars_id[i]].size();
					cur_factor_num_vals_vars.push_back(node_vals[cur_factor_vars_id[i]].size());
				}
				
				// add cpt
				cpt.push_back(Factor(cur_factor_vars_name.size(), cur_factor_vars_name, cur_factor_num_vals_vars, vector<double>(tot_pots, 1.0)));	// 1.0 for smoothing			
					
				// modify adjacency list
				for (int i = 1 ; i < cur_factor_vars_name.size() ; i++)
					adj_list[cur_factor_vars_id[i]].push_back(cur_factor_vars_id[0]);				
			}
			
			if (not getline(infile,line)) break;
		}			
	}
	infile.close();
	init_bn(num_nodes, node_var_names, node_vals, adj_list, cpt);
}

void BayesNet::inference_on_file(string test_filename, string gnd_filename, string out_filename, MarkovNet mn)
{	
	ifstream testfile(test_filename);
	ifstream gndfile(gnd_filename);
	ofstream ofile;
	ofile.open(out_filename);
	string test_line;
	string gnd_line;
	string tempt;
	string tempg;
	double total_missing_vars;
	double total_correct_preds;
	double total_ll;
	int num_queries = 0;
	
	getline(testfile, test_line);
	getline(gndfile, gnd_line);
	
	stringstream ss(test_line);
	
	vector<int> node_id_order;
	for (int i = 0 ; i < num_nodes ; i++)
	{
		ss >> tempt;
		node_id_order.push_back(node_name_to_index[tempt]);		
	}
	
	while(getline(testfile, test_line))
	{
		num_queries++;
		getline(gndfile, gnd_line);
		stringstream sst(test_line);
		stringstream ssg(gnd_line);
		vector<int> query(num_nodes, -1);
		vector<int> gnd_assignment(num_nodes, -1);
		
		for (int i = 0 ; i < num_nodes ; i++)
		{
			sst >> tempt;
			ssg >> tempg;
			
			if (tempt!="?") query[node_id_order[i]] = node_val_to_index[node_id_order[i]][tempt];
			
			gnd_assignment[node_id_order[i]] = node_val_to_index[node_id_order[i]][tempg];			
		}
		
		pair<vector<double>, vector<vector<double> > > inference_results = mn.inference_by_sampling(query, gnd_assignment);
		total_missing_vars += inference_results.first[0];
		total_correct_preds += inference_results.first[1];
		total_ll += inference_results.first[2];
		
		// write marginal probabilities of missing assignments to file
		for (int i = 0 ; i < query.size() ; i++)
			if (query[i] == -1)
			{
				ofile << node_var_names[i] << " ";
				for (int j = 0 ; j < node_vals[i].size() ; j++)
					ofile << node_vals[i][j] << ":" << inference_results.second[i][j] << " ";
				ofile << endl;
			}
		ofile << endl;
	}
	
	testfile.close();
	gndfile.close();
	ofile.close();
	
	cout << "Number of Queries                : " << num_queries << endl;
	cout << "Total Missing Vars               : " << (int) total_missing_vars << endl;
	cout << "Total Correct Preds              : " << (int) total_correct_preds << endl;
	cout << "Percentage Correct               : " << total_correct_preds*100.0/total_missing_vars << endl;
	cout << "Avg Loglikelihood per query      : " << total_ll/num_queries << endl;
	cout << "Avg Loglikelihood per query var  : " << total_ll/total_missing_vars << endl;
}

vector<vector<int> > BayesNet::read_data_file(string filename)
{
	ifstream infile(filename);
	string line;
	string temp;
	vector<vector<int> > data;
	int dat_points = 0;
	
	getline(infile, line);
	stringstream ss(line);
	
	vector<int> node_id_order;
	for (int i = 0 ; i < num_nodes ; i++)
	{
		ss >> temp;
		node_id_order.push_back(node_name_to_index[temp]);		
	}
	
	while (getline(infile, line))
	{
		ss = stringstream(line);
		data.push_back(vector<int>(num_nodes, -1));
		for (int i = 0 ; i < num_nodes ; i++)
		{
			ss >> temp;
			data[dat_points][node_id_order[i]] = node_val_to_index[node_id_order[i]][temp];
		}
		
		dat_points ++;
	}
	infile.close();
	
	return data;
}

void BayesNet::write_params_to_file(string filename, bool write_as_markov_net)
{
	ofstream ofile;
	ofile.open(filename);
	
	ofile << "network unknown {" << endl << "}" << endl;
	
	for (int i = 0 ; i < num_nodes ; i++)
	{
		ofile << "variable " << node_var_names[i] << " {" << endl << "  type discrete [ " << node_vals[i].size() << " ] { " ;
		for (int j = 0 ; j < node_vals[i].size() ; j++)
		{
			ofile << node_vals[i][j];
			if (j!=node_vals[i].size()-1) ofile << ",";
			ofile << " ";
		}		
		
		ofile << "};" << endl << "}" << endl;
	}
	
	for (int i = 0 ; i < CPT.size() ; i++)
	{
		ofile << "probability ( "; 
		
		if (not write_as_markov_net) // write as BN
		{
			ofile << CPT[i].vars_name[CPT[i].num_vars-1] ;
			if (CPT[i].num_vars > 1)
			{				
				ofile << " | ";
				
				for (int j = 0 ; j < CPT[i].num_vars-1 ; j++)
				{
					ofile << CPT[i].vars_name[j];
					if (j!=CPT[i].num_vars-2) ofile << ",";
					ofile << " ";
				}
				
				ofile << ") {" << endl;
			}
			else ofile << " ) {" << endl;	
			
			if (CPT[i].num_vars == 1)
			{
				ofile << "  table ";
				for (int j = 0 ; j < CPT[i].potentials.size() ; j++)
				{
					ofile << CPT[i].potentials[j] ;
					if (j!= CPT[i].potentials.size() - 1) ofile << ", ";
					else ofile << ";" << endl;
				}
			}
			
			else
			{
				vector<int> assignment_to_remaining(CPT[i].num_vars-1, 0);
				int tot_assn_to_rem = accumulate(CPT[i].num_vals_vars.begin(), CPT[i].num_vals_vars.end()-1, 1, multiplies<int>()); // total assignments to the remaining variables
				vector<int> full_assignment;
		    	
				for (int j = 0 ; j < tot_assn_to_rem ; j++)
				{
					ofile << "  (";
					for (int k = 0 ; k < assignment_to_remaining.size() ; k++)
					{
						ofile << node_vals[node_name_to_index[CPT[i].vars_name[k]]][assignment_to_remaining[k]];
						if (k!=assignment_to_remaining.size()-1) ofile << ", ";
						else ofile << ") " ;
					}
					
					for (int k = 0 ; k < CPT[i].num_vals_vars[CPT[i].num_vars-1] ; k++)	// cycle through all values
					{
						full_assignment = assignment_to_remaining;
						full_assignment.push_back(k);
						ofile << CPT[i].pot_at(full_assignment);
						if (k!=CPT[i].num_vals_vars[CPT[i].num_vars-1]-1) ofile << ", ";
						else ofile << ";" << endl;
					}
					
					CPT[i].increment(assignment_to_remaining, vector<int>(CPT[i].num_vals_vars.begin(), CPT[i].num_vals_vars.end()-1));
				}								
			}			
		}
		
		else // write as markov net
		{
			for (int j = 0 ; j < CPT[i].num_vars ; j++)
			{
				ofile << CPT[i].vars_name[j];
				if (j!=CPT[i].num_vars-1) ofile << ",";
				ofile << " ";
			}
			ofile << ") {" << endl;
			
			vector<int> assignment(CPT[i].num_vars, 0);
			int tot_assn = accumulate(CPT[i].num_vals_vars.begin(), CPT[i].num_vals_vars.end(), 1, multiplies<int>());
			for (int j = 0 ; j < tot_assn ; j++)
			{
				ofile << "  (";
				for (int k = 0 ; k < assignment.size() ; k++)
				{
					ofile << node_vals[node_name_to_index[CPT[i].vars_name[k]]][assignment[k]];
					if (k!=assignment.size()-1) ofile << ", ";
					else ofile << ") " ;
				}
				
				ofile << CPT[i].pot_at(assignment) << ";" << endl;
				
				CPT[i].increment(assignment, vector<int>(CPT[i].num_vals_vars.begin(), CPT[i].num_vals_vars.end()));
			}
		}
		ofile << "}" << endl;			
	}
	
	ofile.close();
}

void BayesNet::print()
{
	// printing details
	cout << "Num Nodes : " << num_nodes << endl;
	for (int i = 0 ; i < num_nodes ; i++)
	{
		cout << "Node " << i << "      : " << node_var_names[i] << endl;
		cout << "Node " << i << " vals : " ;
		for (int j = 0 ; j < node_vals[i].size() ; j++)
			cout << node_vals[i][j] << " ";
		cout << endl;
	}
	cout << endl;
	
	// adj list
	cout << "Adjacency List" << endl;
	for (int i = 0 ; i < num_nodes ; i++)
	{
		cout << i << " : " ;
		for (int j = 0 ; j < adj_list[i].size() ; j++)
			cout << adj_list[i][j] << " ";
		cout << endl;
	}
	cout << endl;
	
	// adj list
	cout << "Reverse adjacency List" << endl;
	for (int i = 0 ; i < num_nodes ; i++)
	{
		cout << i << " : " ;
		for (int j = 0 ; j < rev_adj_list[i].size() ; j++)
			cout << rev_adj_list[i][j] << " ";
		cout << endl;
	}
	cout << endl;
	
	cout << "CPTs (" << CPT.size() << " in all)" << endl;
	cout << "==================" << endl << endl;
	for (int i = 0 ; i < CPT.size() ; i++)
		CPT[i].print();
}

int main()
{
	BayesNet bn = BayesNet("../A3-data/insurance.bif");	
	vector<vector<int> > train_data = bn.read_data_file("../A3-data/insurance.dat");
	
	bn.learn_parameters(train_data);	
	
	// bn.write_params_to_file("bn_params.bif",false);
	MarkovNet mn = bn.moralize_bn();
	
	//mn.learn_parameters(train_data, 0.1, 1.0, 0.005, 100);
	bn.CPT = mn.get_factors();	// transfer potentials back for writing to file
	bn.write_params_to_file("../A3-data/mn_params.txt", true);
	
	bn.inference_on_file("../A3-data/insurance_test.dat", "../A3-data/insurance_TrueValues.dat", "../A3-data/bn_mp.mn.out", mn);
	
//	MarkovNet mn = bn.moralize_bn();
//	vector<vector<int> > samples = mn.gibbs_sampler(vector<int>(mn.num_nodes,-1), 3000, 100, 2000, 20000, 0.01);
	
//	vector<vector<int> > train_data = bn.read_data_file("../A3-data/insurance_small.dat");
//	bn.learn_parameters(train_data);	
//	bn.print();


}