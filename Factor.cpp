#include <iostream>
#include <cstdio>
#include <cstdlib>
#include <numeric>
#include <vector>
#include <set>
#include <string>
#include <map>

using namespace std;

#ifndef FACTOR_CPP
	#define FACTOR_CPP

class Factor
{
	// Factor table, stored as a flattened vector<double>
	public:
		Factor(int num_vars, vector<string> vars_name, vector<int> num_vals_vars, vector<double> potentials);
		Factor();	// unity factor
		double pot_at(vector<int> indices) const;
		int flat_index_from_assignment(vector<int>& assignment) const;
		void print();
		
		int num_vars;		
		vector<string> vars_name;
		vector<int> num_vals_vars;
		vector<double> potentials;

		// operations
		Factor operator*(const Factor& f1);
		Factor operator/(const Factor& f1);
		Factor sum_out(string var_name);
		Factor max_out(string var_name);
		void normalize();
		
		// for sampling
		int var_sample(string var_to_sample, map<string,int> assignment_map);
		
		// auxiliary function for incrementing an assignment given the number of values taken by each variable
		void increment(vector<int>& assignment, const vector<int>& num_vals);
};

Factor::Factor(int nv, vector<string> vn, vector<int> nvv, vector<double> pot)
{
	num_vars = nv;
	vars_name = vn;
	num_vals_vars = nvv;
	potentials = pot;
}

Factor::Factor()
{
	// creates a unity factor
	num_vars = 0;
	vars_name = vector<string>();
	num_vals_vars = vector<int>();
	potentials = vector<double>{1};
}

int Factor::flat_index_from_assignment(vector<int>& assignment) const
{
	int flat_index = 0;
	int mult = 1;
	for (int i = assignment.size() - 1 ; i >= 0 ; i--)
	{
		flat_index += mult * assignment[i];
		mult *= num_vals_vars[i];
	}
	
	return flat_index;
}

double Factor::pot_at (vector<int> assignment) const 
{
	int flat_index = flat_index_from_assignment(assignment);

	return potentials[flat_index];
}

void Factor::print()
{
	for (int i = 0 ; i < num_vars ; i++)
		cout << vars_name[i] << '\t';
	cout << endl << endl;

	vector<int> index(num_vars, 0);
	for (int i = 0 ; i < potentials.size() ; i++)
	{
		for (int j = 0 ; j < index.size() ; j++)
			cout << index[j] << '\t';
		cout << pot_at(index) << endl;
		increment(index, this->num_vals_vars);
	}

	cout << endl; 
}

void Factor::increment(vector<int>& assignment, const vector<int>& num_vals)
{
	bool next;
	for (int i = assignment.size() - 1 ; i >= 0 ; i--)
	{
		next = false;
		if ((assignment[i] + 1) == num_vals[i]) next = true;
		assignment[i] = (assignment[i] + 1) % num_vals[i];
		if (not next) break;
	}
}

Factor Factor::operator*(const Factor& f1)
{
	set<string> all_vars;
	all_vars.insert(this->vars_name.begin(), this->vars_name.end());
	all_vars.insert(f1.vars_name.begin(), f1.vars_name.end());
	vector<string> vars_union(all_vars.begin(), all_vars.end());
	vector<int> new_num_vals_vars;
	vector<double> new_pots;	

	// constructing new_num_vals_vars (inefficient)
	vector<string>::iterator it;
	for (int i = 0 ; i<vars_union.size() ; i++)
	{
		it = find(this->vars_name.begin(), this->vars_name.end(), vars_union[i]);
		if (it != this->vars_name.end())
		{
			new_num_vals_vars.push_back(this->num_vals_vars[it-this->vars_name.begin()]);
			continue;
		}
		else
		{			
			new_num_vals_vars.push_back(f1.num_vals_vars[find(f1.vars_name.begin(), f1.vars_name.end(), vars_union[i])-f1.vars_name.begin()]);
		}
	}

	map<string, int> var_name_to_pos;
	for (int i = 0 ; i < vars_union.size() ; i++)
		var_name_to_pos.insert(pair<string, int>(vars_union[i], i));

	// element wise multiplication
	vector<int> index(vars_union.size(), 0);
	for (int i = 0 ; i < accumulate(new_num_vals_vars.begin(), new_num_vals_vars.end(), 1, multiplies<int>()) ; i++)
	{
		vector<int> self_ind;
		for (int j = 0 ; j < this->vars_name.size() ; j++)
			self_ind.push_back(index[var_name_to_pos[this->vars_name[j]]]);
		vector<int> f1_ind;
		for (int j = 0 ; j < f1.vars_name.size() ; j++)
			f1_ind.push_back(index[var_name_to_pos[f1.vars_name[j]]]);

		new_pots.push_back(this->pot_at(self_ind)*f1.pot_at(f1_ind));

		increment(index, new_num_vals_vars);
	}

	return Factor(vars_union.size(), vars_union, new_num_vals_vars, new_pots);
}

Factor Factor::operator/(const Factor& f1)
{
	// scope(this) is >= scope(f1)
	vector<double> new_pots;

	vector<int> vars_name_intersect_indices;
	for (int i = 0 ; i < f1.vars_name.size() ; i++)
		for (int j = 0 ; j < this->vars_name.size() ; j++)
			if (f1.vars_name[i] == this->vars_name[j])
			{
				vars_name_intersect_indices.push_back(j);
				break;
			}

	vector<int> index(this->vars_name.size(), 0);
	for (int i = 0 ; i < this->potentials.size() ; i++)
	{
		vector<int> f1_index;
		for (int j = 0 ; j < vars_name_intersect_indices.size() ; j++)
			f1_index.push_back(index[vars_name_intersect_indices[j]]);

		if (f1.pot_at(f1_index) == 0)
		{
			new_pots.push_back(0);
		}

		else
		{
			new_pots.push_back(potentials[i]/f1.pot_at(f1_index));
		}

		increment(index, this->num_vals_vars);
	}

	return Factor(this->num_vars, this->vars_name, this->num_vals_vars, new_pots);
}

Factor Factor::sum_out(string var_name)
{
	// sum_out only if var_name present in scope! 
	// came across some cases of BP in clique trees where initially scopes of nodes are not complete initially
	int pos = -1;
	vector<string> new_vars_name;
	vector<int> new_num_vals_vars;
	for (int i = 0 ; i<vars_name.size() ; i++)
		if (vars_name[i]==var_name)
		{
			pos = i;
		}
		else
		{
			new_vars_name.push_back(vars_name[i]);
			new_num_vals_vars.push_back(num_vals_vars[i]);
		}

	if (pos==-1) return *this;	// can't sum out something not there!

	vector<double> new_pots;

	vector<int> index(new_vars_name.size(), 0);
	vector<int> mod_index;
	for (int i = 0 ; i < accumulate(new_num_vals_vars.begin(), new_num_vals_vars.end(), 1, multiplies<int>()) ; i++)
	{
		double sum = 0;
		for (int j = 0 ; j < num_vals_vars[pos] ; j++)
		{
			mod_index = index;
			mod_index.insert(mod_index.begin() + pos, 1, j);
			sum += pot_at(mod_index);
		}
		new_pots.push_back(sum);

		increment(index, new_num_vals_vars);
	}
	
	return Factor(new_vars_name.size(), new_vars_name, new_num_vals_vars, new_pots);
}

Factor Factor::max_out(string var_name)
{
	int pos = -1;
	vector<string> new_vars_name;
	vector<int> new_num_vals_vars;
	for (int i = 0 ; i<vars_name.size() ; i++)
		if (vars_name[i]==var_name)
		{
			pos = i;
		}
		else
		{
			new_vars_name.push_back(vars_name[i]);
			new_num_vals_vars.push_back(num_vals_vars[i]);
		}

	if (pos==-1) return *this;	 // can't sum out something not there!

	vector<double> new_pots;

	vector<int> index(new_vars_name.size(), 0);
	vector<int> mod_index;
	for (int i = 0 ; i < accumulate(new_num_vals_vars.begin(), new_num_vals_vars.end(), 1, multiplies<int>()) ; i++)
	{
		double cur_max = -1;
		for (int j = 0 ; j < num_vals_vars[pos] ; j++)
		{
			mod_index = index;
			mod_index.insert(mod_index.begin() + pos, 1, j);
			if (pot_at(mod_index) > cur_max) cur_max = pot_at(mod_index);		
		}
		new_pots.push_back(cur_max);

		increment(index, new_num_vals_vars);
	}
	
	return Factor(new_vars_name.size(), new_vars_name, new_num_vals_vars, new_pots);
}

void Factor::normalize()
{
	double sum = accumulate(potentials.begin(), potentials.end(), 0.0);	

	for (int i = 0 ; i<potentials.size() ; i++)
		potentials[i] = potentials[i]/sum;
}

int Factor::var_sample(string var_to_sample, map<string,int> assignment_map)
{
	// in the current factor, given assignments to all variables other than var_to_sample, sample var_to_sample from the conditional distribution
	int var_to_sample_id;
	for (int i = 0 ; i < num_vars ; i++)
		if (vars_name[i] == var_to_sample)
		{
			var_to_sample_id = i;
			break;
		}
	
	vector<double> prob(num_vals_vars[var_to_sample_id], 0.0);		// will be unnormalized initially
	vector<int> assignment(num_vars, -1);
	for (int i = 0 ; i < num_vars ; i++)
		if (i != var_to_sample_id)
		{
			assignment[i] = assignment_map[vars_name[i]];
		}
	
	double sum = 0.0;
	for (int i = 0 ; i < num_vals_vars[var_to_sample_id] ; i++)
	{
		assignment[var_to_sample_id] = i;
		prob[i] = pot_at(assignment);
		sum += prob[i];
	}
	
	double r = ((double) rand() / (RAND_MAX));
	// normalize and make cumulative prob
	for (int i = 0 ; i < num_vals_vars[var_to_sample_id] ; i++)
	{
		prob[i] /= sum;
		if (i>0) prob[i] += prob[i-1];	// cumulative
		if (r <= prob[i]) return i;
	}
	
	cout << "Sampling Error" << endl;
	this->print();
	// for safety
	return -1;
}

#endif