#include <iostream>
#include <cstdlib>
#include <fstream>
#include <string>
#include <sstream>
#include <vector>
#include <set>
#include <map>
#include <algorithm>
#include <math.h>

#include "MarkovNet.cpp"
#include "FactorGraph.cpp"
#include "Factor.cpp"

class OCR
{
	public:
		OCR(int num_images, vector<string> chars, double skip_factor, double pair_skip_factor, string ocr_factors_filename, string trans_factors_filename);		
		void classify_file(string input_filename, string output_filename, string gnd_filename, int mode, bool margin_sum, bool trans, bool skip, bool pair_skip);
		pair<pair<string, double>, pair<string, double> > classify_img_pair(vector<int> imgs1, vector<int> imgs2, vector<int> gnd_assignment1, vector<int> gnd_assignment2, int mode, bool margin_sum, bool trans, bool skip, bool pair_skip);
		MarkovNet gen_pair_mn(vector<int> imgs1, vector<int> imgs2, bool trans, bool skip, bool pair_skip);	// make private later?		
	private:
		void load_phi_o(string filename);
		void load_phi_t(string filename);

		Factor gen_ocr_factor(string var_name, int img);
		Factor gen_trans_factor(string var_name1, string var_name2);
		Factor gen_pair_factor(string var_name1, string var_name2, bool pair_skip);	
		
		

		int num_images;
		int dict_size;
		map<string, int> char_int_map;
		map<int, string> int_char_map;
		vector<vector<double> > ocr_factors;
		vector<vector<double> > trans_factors;
		double skip_factor;
		double pair_skip_factor;
};

OCR::OCR(int ni, vector<string> chars, double sf, double psf, string ocr_factors_filename, string trans_factors_filename)
{
	num_images = ni;
	dict_size = chars.size();
	skip_factor = sf;
	pair_skip_factor = psf;
	ocr_factors = vector<vector<double> >(num_images, vector<double>(dict_size, 0));
	trans_factors = vector<vector<double> >(dict_size, vector<double>(dict_size, 0));
	
	for (int i = 0 ; i<chars.size(); i++)
	{
		char_int_map.insert(pair<string,int>(chars[i], i));
		int_char_map.insert(pair<int,string>(i, chars[i]));
	}
	
	load_phi_o(ocr_factors_filename);
	load_phi_t(trans_factors_filename);
}

void OCR::load_phi_o(string filename)
{
	string line;
	ifstream infile(filename);
	if (infile.is_open())
	{
		while (getline(infile,line))
	    {
			stringstream ss(line);
			int i; ss >> i; ss.ignore();	// ignore tab
			string s; ss >> s; ss.ignore();
			double f; ss >> f;

			ocr_factors[i][char_int_map[s]] = f;						
		}
		infile.close();
	}
}

void OCR::load_phi_t(string filename)
{
	string line;
	ifstream infile(filename);
	if (infile.is_open())
	{
		while (getline(infile,line))
	    {
			stringstream ss(line);
			string s1; ss >> s1; ss.ignore();	// ignore tab
			string s2; ss >> s2; ss.ignore();
			double f; ss >> f;
			
			trans_factors[char_int_map[s1]][char_int_map[s2]] = f;			
		}
		infile.close();
	}
}

Factor OCR::gen_ocr_factor(string var_name, int img)
{
	// var_name.size() == 1
	return Factor(1, vector<string>{var_name}, vector<int>{dict_size}, ocr_factors[img]);
}
		
Factor OCR::gen_trans_factor(string var_name1, string var_name2)
{
	// var_names.size() == 2
	vector<double> potentials;

	for (int i = 0 ; i < dict_size ; i++)
		potentials.insert(potentials.end(), trans_factors[i].begin(), trans_factors[i].end());

	return Factor(2, vector<string>{var_name1, var_name2}, vector<int>{dict_size, dict_size}, potentials);
}

Factor OCR::gen_pair_factor(string var_name1, string var_name2, bool pair_skip)
{
	// var_names.size() == 2
	// if pair_skip => pair_skip factor, else normal skip factor

	vector<double> potentials;

	for (int i = 0 ; i < dict_size ; i++)
		for (int j = 0 ; j < dict_size ; j++)
		{
			if (i==j and not pair_skip) potentials.push_back(skip_factor);
			else if (i==j and pair_skip) potentials.push_back(pair_skip_factor);
			else potentials.push_back(1.0);
		}

	return Factor(2, vector<string>{var_name1, var_name2}, vector<int>{dict_size, dict_size}, potentials);
}


MarkovNet OCR::gen_pair_mn(vector<int> imgs1, vector<int> imgs2, bool trans, bool skip, bool pair_skip)
{
	vector<string> node_var_names;
	vector<vector<int> > adj_list(imgs1.size()+imgs2.size(), vector<int>());
	vector<Factor> factors;

	for (int i = 0 ; i < imgs1.size() ; i++)
		node_var_names.push_back(string("w1_")+to_string(i));
	for (int i = 0 ; i < imgs2.size() ; i++)
		node_var_names.push_back(string("w2_")+to_string(i));

	// ocr factors
	for (int i = 0 ; i < imgs1.size() ; i++)
		factors.push_back(gen_ocr_factor(node_var_names[i], imgs1[i]));
	for (int i = 0 ; i < imgs2.size() ; i++)
		factors.push_back(gen_ocr_factor(node_var_names[i+imgs1.size()], imgs2[i]));

	// trans factors
	if (trans)
	{
		for (int i = 0 ; i < imgs1.size() - 1 ; i++)
		{
			factors.push_back(gen_trans_factor(node_var_names[i], node_var_names[i+1]));
			adj_list[i].push_back(i+1);
			adj_list[i+1].push_back(i);
		}
		for (int i = 0 ; i < imgs2.size() - 1 ; i++)
		{
			factors.push_back(gen_trans_factor(node_var_names[i+imgs1.size()], node_var_names[i+1+imgs1.size()]));
			adj_list[i+imgs1.size()].push_back(i+1+imgs1.size());
			adj_list[i+imgs1.size()+1].push_back(i+imgs1.size());
		}
	}

	// skip factors
	if (skip)
	{
		for (int i = 0 ; i < imgs1.size() ; i++)
			for (int j = i+1 ; j < imgs1.size() ; j++)
				if (imgs1[i]==imgs1[j])
				{
					factors.push_back(gen_pair_factor(node_var_names[i], node_var_names[j], false));
					adj_list[i].push_back(j);
					adj_list[j].push_back(i);
				}

		for (int i = 0 ; i < imgs2.size() ; i++)
			for (int j = i+1 ; j < imgs2.size() ; j++)
				if (imgs2[i]==imgs2[j])
				{
					factors.push_back(gen_pair_factor(node_var_names[i + imgs1.size()], node_var_names[j + imgs1.size()], false));
					adj_list[i + imgs1.size()].push_back(j + imgs1.size());
					adj_list[j + imgs1.size()].push_back(i + imgs1.size());
				}
	}

	// pair skip factors
	if (pair_skip)
	{
		for (int i = 0 ; i < imgs1.size() ; i++)
			for (int j = 0 ; j < imgs2.size() ; j++)
				if (imgs1[i] == imgs2[j])
				{
					factors.push_back(gen_pair_factor(node_var_names[i], node_var_names[j + imgs1.size()], true));
					adj_list[i].push_back(j + imgs1.size());
					adj_list[j + imgs1.size()].push_back(i);
				}
	}

	return MarkovNet(node_var_names.size(), node_var_names, vector<int>(node_var_names.size(), dict_size), adj_list, factors);
}

pair<pair<string, double>, pair<string, double> > OCR::classify_img_pair(vector<int> imgs1, vector<int> imgs2, vector<int> gnd_assignment1, vector<int> gnd_assignment2, int mode, bool margin_sum, bool trans, bool skip, bool pair_skip)
{
	// modes
	// 1: Message Passing
	// 2: Loopy BP
	// 3: Gibbs Sampling
	
	MarkovNet mn = gen_pair_mn(imgs1, imgs2, trans, skip, pair_skip);

	FactorGraph fg;
	vector<int> pred_assignment;
	vector<int> gnd_assignment = gnd_assignment1;
	gnd_assignment.insert(gnd_assignment.end(), gnd_assignment2.begin(), gnd_assignment2.end());
	vector<double> gnd_loglikelihood(imgs1.size() + imgs2.size(), 0.0);

	if (mode==1)
	{	  	 	  
	  fg = mn.gen_clique_tree(mn.min_fill_ve_order());
	  
	  if (margin_sum) fg.MessagePassing(0, &Factor::sum_out);	  
	  else fg.MessagePassing(0, &Factor::max_out);	  	  
	}
	
	else if (mode==2)	// Loopy BeliefProp
	{
		fg = mn.gen_bethe_cluster_graph();
		if (margin_sum) fg.BeliefProp(0.001, 1000, &Factor::sum_out, true);
		else fg.BeliefProp(0.001, 1000, &Factor::max_out, true);
	}
	
	else if (mode==3) // Gibbs Sampling
	{
		if (margin_sum)
		{
			vector<vector<int> > samples = mn.gibbs_sampler(vector<int>(mn.num_nodes, -1), 5000, 100, 5000, 15000, 0.1);
			pair<vector<int>, vector<vector<double> > > dist = mn.marginal_prob_dist_from_samples(samples);
			pred_assignment = dist.first;
			gnd_loglikelihood = mn.marginal_likelihood(gnd_assignment, dist.second);
		}
		else cout << "Gibbs Sampling not implemented for MAP" << endl;			
	}
	
	if (mode==1 or mode==2)
	{
		if (margin_sum) 
		{
			pred_assignment = fg.max_marginal_assignment(&Factor::sum_out).first;
			gnd_loglikelihood = fg.marginal_likelihood(false, gnd_assignment);
		}
		else
		{ 
			pred_assignment = fg.max_marginal_assignment(&Factor::max_out).first;		
			gnd_loglikelihood = fg.marginal_likelihood(true, gnd_assignment);				
		}
	}
	
	string w1 = "";
	string w2 = "";
	double ll1 = 0.0;
	double ll2 = 0.0;

	for (int i = 0 ; i < imgs1.size() ; i++)
	{
		w1 += int_char_map[pred_assignment[i]];
		ll1 += gnd_loglikelihood[i];
	}
	for (int j = 0 ; j < imgs2.size() ; j++)
	{
		w2 += int_char_map[pred_assignment[j + imgs1.size()]];
		ll2 += gnd_loglikelihood[j + imgs1.size()];
	}

	return pair<pair<string, double>, pair<string, double> >(pair<string, double>(w1, ll1), pair<string, double>(w2, ll2));
}

void OCR::classify_file(string input_filename, string output_filename, string gnd_filename, int mode, bool margin_sum, bool trans, bool skip, bool pair_skip)
{
	// modes
	// 1: Message Passing
	// 2: Loopy BP
	// 3: Gibbs Sampling
	
	string inp_line;
	string gnd_line;
	ifstream infile(input_filename);
	ifstream gndfile(gnd_filename);
	ofstream ofile;
	ofile.open(output_filename);
	double avgLogProb = 0.0;
	int i = 0;
	
	if (infile.is_open())
	{
		while (getline(infile,inp_line))
	    {	
			getline(gndfile,gnd_line);			
	    	if (not inp_line.empty())
	    	{		
				// cout << i << endl;				
				vector<int> cur_imgs1;
				vector<int> cur_imgs2;
				vector<int> gnd_assngmt1;
				vector<int> gnd_assngmt2;
				
				stringstream ss1(inp_line);
				int n;
				
				while (ss1 >> n)
				{
					cur_imgs1.push_back(n);
					if (ss1.peek() == '\t') ss1.ignore();				
				}
					
				for (int j = 0 ; j<gnd_line.size() ; j++) gnd_assngmt1.push_back(char_int_map[string(1,gnd_line.at(j))]);
				
				getline(infile,inp_line);
				stringstream ss2(inp_line);			
				
				while (ss2 >> n)
				{
					cur_imgs2.push_back(n);
					if (ss2.peek() == '\t') ss2.ignore();				
				}
				
				getline(gndfile,gnd_line);			
				for (int j = 0 ; j<gnd_line.size() ; j++) gnd_assngmt2.push_back(char_int_map[string(1,gnd_line.at(j))]);

				pair<pair<string, double>, pair<string, double> > pred = classify_img_pair(cur_imgs1, cur_imgs2, gnd_assngmt1, gnd_assngmt2, mode, margin_sum, trans, skip, pair_skip);
				ofile << pred.first.first << endl << pred.second.first << endl << endl;

				avgLogProb += pred.first.second + pred.second.second;
				i += 2;
			}
		}
		infile.close();
	}
	ofile.close();	
	avgLogProb = avgLogProb/i;
	cout << "Average Log Likelihood (as defined for both cases in the assignment) : " << avgLogProb << endl;
}

void print_stats(string ref_file, string pred_file)
{
	string line;
	ifstream f1(ref_file);
	ifstream f2(pred_file);
	
	vector<string> words1;
	vector<string> words2;
	
	int total_chars = 0;
	int match_chars = 0;
	int total_words = 0;
	int match_words = 0;
	
	if (f1.is_open())
	{
		while (getline(f1, line))
			if (not line.empty())
				words1.push_back(line);
		f1.close();
	}
	
	if (f2.is_open())
	{
		while (getline(f2, line))
			if (not line.empty())
				words2.push_back(line);

		f2.close();
	}
	
	for (int i = 0 ; i<words1.size() ; i++)
	{
		total_words += 1;
		if (words1[i]==words2[i]) match_words += 1;
		// cout << words1[i] << " " << words2[i] << "\n";
		
		total_chars += words1[i].size();
		
		for (int j = 0 ; j<words1[i].size() ; j++)
			if (words1[i][j]==words2[i][j])
				match_chars += 1;
	}
	
	cout << "correct words/total words : " << match_words << "/" << total_words << " (" << (double)100.0*match_words/(double)total_words << "%)" << "\n";
	cout << "correct chars/total chars : " << match_chars << "/" << total_chars << " (" << (double)100.0*match_chars/(double)total_chars << "%)" << "\n\n";
}


int main()
{	
	
	OCR ocr = OCR(1000, vector<string>{"d","o","i","r","a","h","t","n","s","e"}, 5.0, 5.0, "../OCRdataset-2/potentials/ocr.dat", "../OCRdataset-2/potentials/trans.dat");
	
	//pair<pair<string, double>, pair<string, double> > pred = ocr.classify_img_pair(vector<int>{592,688,240,592}, vector<int>{999,773,575,592,721,960}, vector<int>{0,0,0,0}, vector<int>{0,0,0,0,0,0}, 3, true, true, true, true);
	//cout << pred.first.first << " " << pred.second.first << endl;
	//MarkovNet mn = ocr.gen_pair_mn(vector<int>{592,688,240,592}, vector<int>{999,773,575,592,721,960}, true, true, true);
	
	//ocr.classify_file("../OCRdataset-2/data/data-tree.dat", "../OCRdataset-2/data/pred.dat","../OCRdataset-2/data/truth-tree.dat", 2, true, true, true, true);
	//print_stats("../OCRdataset-2/data/truth-tree.dat", "../OCRdataset-2/data/pred.dat");
	
	//ocr.classify_file("../OCRdataset-2/data/data-treeWS.dat", "../OCRdataset-2/data/pred.dat","../OCRdataset-2/data/truth-treeWS.dat", 2, true, true, true, true);
	//print_stats("../OCRdataset-2/data/truth-treeWS.dat", "../OCRdataset-2/data/pred.dat");

	//ocr.classify_file("../OCRdataset-2/data/data-loops.dat", "../OCRdataset-2/data/pred.dat","../OCRdataset-2/data/truth-loops.dat", 2, true, true, true, true);
	//print_stats("../OCRdataset-2/data/truth-loops.dat", "../OCRdataset-2/data/pred.dat");

	ocr.classify_file("../OCRdataset-2/data/data-loopsWS.dat", "../OCRdataset-2/data/pred.dat","../OCRdataset-2/data/truth-loopsWS.dat", 3, true, true, true, true);
	print_stats("../OCRdataset-2/data/truth-loopsWS.dat", "../OCRdataset-2/data/pred.dat");
	
	/*
	string sA = string("A");
	string sB = string("B");
	string sC = string("C");
	string sD = string("D");
	string sE = string("E");
	string sF = string("F");
	string sG = string("G");
	string sH = string("H");

	
	Factor phi_A(1, vector<string>{sA}, vector<int>{2}, vector<double>{1,1});
	Factor phi_B(1, vector<string>{sB}, vector<int>{2}, vector<double>{2,1});
	Factor sob = phi_B.sum_out(sB);
	sob.print();
	Factor phi_C(1, vector<string>{sC}, vector<int>{2}, vector<double>{1,1});
	Factor phi_D(1, vector<string>{sD}, vector<int>{2}, vector<double>{1,2});
	Factor phi_AB(2, vector<string>{sA, sB}, vector<int>{2,2}, vector<double>{3,2,1,3});
	Factor phi_BC(2, vector<string>{sB, sC}, vector<int>{2,2}, vector<double>{1,1,1,1});
	Factor phi_CD(2, vector<string>{sC, sD}, vector<int>{2,2}, vector<double>{5,1,2,5});
	Factor phi_DA(2, vector<string>{sD, sA}, vector<int>{2,2}, vector<double>{1,3,3,1});

	Factor phi_ab(2, vector<string>{sA, sB}, vector<int>{2,2}, vector<double>{1,1,1,1});
	Factor phi_bc(2, vector<string>{sB, sC}, vector<int>{2,2}, vector<double>{1,1,1,1});
	Factor phi_cd(2, vector<string>{sC, sD}, vector<int>{2,2}, vector<double>{1,1,1,1});
	Factor phi_da(2, vector<string>{sD, sA}, vector<int>{2,2}, vector<double>{1,1,1,1});
	
	MarkovNet mn = MarkovNet(4, vector<string>{sA,sB,sC,sD}, vector<int>{2,2,2,2}, vector<vector<int> >{vector<int>{1,3}, vector<int>{0,2}, vector<int>{1,3},vector<int>{0,2}}, vector<Factor>{phi_AB, phi_BC, phi_CD, phi_DA});
	mn.print(true);
	
	vector<vector<int> > samples = mn.gibbs_sampler(vector<int>(mn.num_nodes, -1), 3000, 100, 1000, 15000, 0.01);
	cout << samples.size() << endl;
	
	MarkovNet mn2 = MarkovNet(4, vector<string>{sA,sB,sC,sD}, vector<int>{2,2,2,2}, vector<vector<int> >{vector<int>{1,3}, vector<int>{0,2}, vector<int>{1,3},vector<int>{0,2}}, vector<Factor>{phi_ab, phi_bc, phi_cd, phi_da});
	mn2.learn_parameters(samples, 0.01, 0, 0.001);
	mn2.print(true);
	

	Factor phi_ABC(3, vector<string>{sA, sB, sC}, vector<int>{2,2,2}, vector<double>{});
	Factor phi_CDE(3, vector<string>{sC, sD, sE}, vector<int>{2,2,2}, vector<double>{});
	Factor phi_BCE(3, vector<string>{sB, sC, sE}, vector<int>{2,2,2}, vector<double>{});
	Factor phi_BEG(3, vector<string>{sE, sB, sG}, vector<int>{2,2,2}, vector<double>{});
	Factor phi_BFG(3, vector<string>{sF, sB, sG}, vector<int>{2,2,2}, vector<double>{});
	Factor phi_GEH(3, vector<string>{sG, sE, sH}, vector<int>{2,2,2}, vector<double>{});

	FactorGraph fgr(4, vector<set<string> >{set<string>{sA, sB, sD}, set<string>{sB, sC, sD}, set<string>{sC, sD}, set<string>{sD}}, vector<vector<int> >{vector<int>{1}, vector<int>{0,2}, vector<int>{1,3}, vector<int>{2}}, vector<Factor>{phi_A, phi_B, phi_C, phi_D, phi_AB, phi_BC, phi_CD, phi_DA}, vector<vector<int> >{vector<int>{0,4,7}, vector<int>{1,5}, vector<int>{2,6}, vector<int>{3}});
	fgr.print();
	fgr.MessagePassing(3);

	for (int i = 0 ; i < fgr.num_nodes ; i++)
		{fgr.node_marginals[i].normalize(); fgr.node_marginals[i].print();}

	fg.BeliefProp(0.05, 1000);

	for (int i = 0 ; i < fg.num_nodes ; i++)
	{
		fg.node_marginals[i].normalize();
		fg.node_marginals[i].print();
	}

	FactorGraph bcgr(8, vector<set<string> >{set<string>{sA, sB}, set<string>{sB, sC}, set<string>{sC, sD}, set<string>{sD, sA}, set<string>{sA}, set<string>{sB}, set<string>{sC}, set<string>{sD}}, vector<vector<int> >{vector<int>{4,5}, vector<int>{5,6}, vector<int>{6,7}, vector<int>{7, 4}, vector<int>{0,3}, vector<int>{0,1}, vector<int>{1,2}, vector<int>{2,3}}, vector<Factor>{phi_A, phi_B, phi_C, phi_D, phi_AB, phi_BC, phi_CD, phi_DA}, vector<vector<int> >{vector<int>{0,4}, vector<int>{1,5}, vector<int>{2,6}, vector<int>{3,7}, vector<int>(), vector<int>(), vector<int>(), vector<int>()});
	bcgr.print();
	bcgr.BeliefProp(0.001, 1000);
	for (int i = 0 ; i < bcgr.num_nodes ; i++)
	{
		bcgr.node_marginals[i].normalize();
		bcgr.node_marginals[i].print();
	}

	MarkovNet mn = MarkovNet(4, vector<string>{sA,sB,sC,sD}, vector<vector<int> >{vector<int>{1,3}, vector<int>{0,2}, vector<int>{1,3},vector<int>{0,2}}, vector<Factor>{phi_A, phi_B, phi_C, phi_D, phi_AB, phi_BC, phi_CD, phi_DA});
	mn.print(true);
	// MarkovNet mn = MarkovNet(8, vector<string>{sA,sB,sC,sD,sE,sF,sG,sH}, vector<vector<int> >{vector<int>{1,2}, vector<int>{0,2,4,5,6}, vector<int>{0,1,3,4}, vector<int>{2,4}, vector<int>{1,2,3,6,7}, vector<int>{1,6}, vector<int>{1,4,5,7}, vector<int>{4,6}}, vector<Factor>{phi_ABC, phi_CDE, phi_BCE, phi_BEG, phi_BFG, phi_GEH});

	FactorGraph bcg = mn.gen_bethe_cluster_graph();
	bcg.print();
	bcg.BeliefProp(0.001,1000);
	for (int i = 0 ; i < bcg.num_nodes ; i++)
	{
		bcg.node_marginals[i].normalize();
		bcg.node_marginals[i].print();
	}


	vector<int> order = mn.min_fill_ve_order();
	cout << "Min Fill Order : ";
	for (int i = 0 ; i <order.size() ; i++)
		cout << order[i] << " ";
	cout <<endl << endl;

	FactorGraph fg = mn.gen_clique_tree(mn.min_fill_ve_order());
	fg.print();
	fg.MessagePassing(0);
	for (int i = 0 ; i < fg.num_nodes ; i++)
		{fg.node_marginals[i].normalize(); fg.node_marginals[i].print();}
	*/
}