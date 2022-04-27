#include <iostream>
#include <fstream>
#include <set>
#include <ctime>
#include <assert.h>
#include <map>
#include "utils.h"
#include "kg.h"
#include "rule.h"
#include "train.h"
using namespace std;

int test0();
int test1();
int test_train0();
int test_grad();
int test_forward();
void test_accuracy();
void test_vfrom();

int main()
{
    test_train0();
    // test_accuracy();
    // test_grad();
    // test_forward();

    return 0;
}

int test_train0()
{
    clock_t start,end;
    srand(time(0));
    int v_from, v_to;

    KnowledgeGraph KG;
    // KG.load_data((char*)"../data/umls/train.txt");
    // KG.load_data((char*)"../data/kinship/train.txt");
    // KG.load_data((char*)"../data/wn18rr/train.txt");
    KG.load_data((char*)"../data/FB15k-237/train.txt");
    Graph G(KG.n_entity, KG.n_relation_with_reverse());
    for (auto it = KG.triplets.begin(); it != KG.triplets.end(); it++)
    {
        G.add_edge(it->h, it->t, it->r);
    }
    v_from = rand() % G.n_vertex;
    v_to = rand() % G.n_vertex;

    RuleTrainer x = RuleTrainer(3, &G);
    x.negative_sample(3);
    // x.negative_sample_all();

    vector<Rule> *rules = new vector<Rule>[G.n_edge_type];

    start = clock();
    x.train(100, 10, rules);
    // x.train_one_rule(70, 10);
    // cout<<x.pos[0][0].size()<<" "<<x.neg[0][0].size()<<endl;
    end = clock();
	double endtime=(double)(end-start)/CLOCKS_PER_SEC;
    std::cout<<endtime<<std::endl;

    ofstream fout("output.txt");
    fout<<"[";
    for (int i = 0; i < G.n_edge_type; i++)
    {
        fout<<"[";
        for (int j = 0; j < rules[i].size(); j++)
        {
            fout<<"[";
            fout<<"\""<<KG.id2r[rules[i][j].head]<<"\",";
            for (int k = 0; k < 3; k++)
                fout<<"\""<<KG.id2r[rules[i][j].r_body[k]]<<"\",";
            fout<<"],\n";
        }
        fout<<"],";
    }
    fout<<"]";
    fout.close();


    // cout<<KG.id2e[108]<<" "<<KG.id2r[6]<<" "<<KG.id2e[80]<<endl;

    cout<<endl;

    delete[] rules;
    return 0;
}

void test_accuracy()
{
    int rule_len = 3;
    KnowledgeGraph KG;
    KG.load_data((char*)"../data/umls/train.txt");
    // KG.load_data((char*)"../data/kinship/train.txt");
    // KG.load_data((char*)"../data/testdata.txt");
    // KG.load_data((char*)"../data/wn18rr/train.txt");
    // KG.load_data((char*)"../data/FB15k-237/train.txt");
    Graph G(KG.n_entity, KG.n_relation_with_reverse());
    for (auto it = KG.triplets.begin(); it != KG.triplets.end(); it++)
    {
        G.add_edge(it->h, it->t, it->r);
    }
    RuleTrainer x = RuleTrainer(rule_len, &G);
    x.negative_sample_all();
    Rule rule;
    rule.head = KG.r2id["affects"];
    rule.r_body.push_back(KG.r2id["rev-manifestation_of"]);
    rule.r_body.push_back(KG.r2id["affects"]);
    rule.r_body.push_back(KG.r2id["rev-process_of"]);
    // x.reweight(&rule);

    real_t a[G.n_vertex][G.n_vertex];
    for (int i = 0; i < G.n_vertex; i++)
    {
        for (int j = 0; j < G.n_vertex; j++)
        {
            a[i][j] = 0;
        }
    }
    for (auto ptrvfrom = x.vfrom[rule.head].begin(); ptrvfrom != x.vfrom[rule.head].end(); ptrvfrom++)
    {
        for (int i = 0; i < x.vto[rule.head][*ptrvfrom].size(); i++)
        {
            a[*ptrvfrom][x.vto[rule.head][*ptrvfrom][i]] += x.weight[rule.head][*ptrvfrom][i];
        }
    }

    real_t sum = 0;
    int cnt = 0;
    for (int i = 0; i < G.n_vertex; i++)
    {
        for (int j = 0; j < G.n_vertex; j++)
        {
            cout<<a[i][j]<<" ";
            sum += a[i][j];
            if (a[i][j] > 0) cnt++;
        }
        cout<<endl;
    }
    cout<<endl;
    cout<<"sum: "<<sum<<endl<<"cnt: "<<cnt<<endl;
    
}

int test_forward()
{
    int rule_len = 3;
    KnowledgeGraph KG;
    // KG.load_data((char*)"../data/umls/train.txt");
    KG.load_data((char*)"../data/kinship/train.txt");
    // KG.load_data((char*)"../data/testdata.txt");
    // KG.load_data((char*)"../data/wn18rr/train.txt");
    // KG.load_data((char*)"../data/FB15k-237/train.txt");
    Graph G(KG.n_entity, KG.n_relation_with_reverse());
    for (auto it = KG.triplets.begin(); it != KG.triplets.end(); it++)
    {
        G.add_edge(it->h, it->t, it->r);
    }

    dRule r = dRule(rule_len, &G, false);
    r.forward_R();
    vector<int> vto;
    // for (int i = 0; i < G.n_vertex; i++) vto.push_back(i);
    vto.push_back(0);
    auto V = r.forward(0, vto);
    V = r.forward(0, vto);
    for (int i = 0; i < G.n_vertex; i++)
    {
        cout<<r.V[rule_len][i].value<<" ";
    }
    cout<<"\n\n";

    return 0;
}

int test_grad()
{
    int rule_len = 3;
    KnowledgeGraph KG;
    KG.load_data((char*)"../data/umls/train.txt");
    // KG.load_data((char*)"../data/kinship/train.txt");
    // KG.load_data((char*)"../data/testdata.txt");
    // KG.load_data((char*)"../data/wn18rr/train.txt");
    // KG.load_data((char*)"../data/FB15k-237/train.txt");
    Graph G(KG.n_entity, KG.n_relation_with_reverse());
    for (auto it = KG.triplets.begin(); it != KG.triplets.end(); it++)
    {
        G.add_edge(it->h, it->t, it->r);
    }

    RuleTrainer x = RuleTrainer(rule_len, &G);
    x.negative_sample_all();
    x.train_one_rule(KG.r2id["affects"], 1);


    cout<<"--------------------------\n";



    return 0;
}

int test0()
{
    srand(time(0));
    clock_t start,end;
    KnowledgeGraph KG;
    KG.load_data((char*)"../data/umls/train.txt");
    // KG.load_data((char*)"../data/wn18rr/train.txt");
    // KG.load_data((char*)"../data/FB15k-237/train.txt");
    Graph G(KG.n_entity, KG.n_relation_with_reverse());
    for (auto it = KG.triplets.begin(); it != KG.triplets.end(); it++)
    {
        G.add_edge(it->h, it->t, it->r);
    }

    int batch = 1000;
    std::vector<double> R;
    for (int i = 0; i < KG.n_relation_with_reverse(); i++)
    {
        double f = randn();
        R.push_back(f * f + 0.1);
    }

    std::vector<double> status[4];
    for (int i = 0; i < KG.n_entity; i++)
    {
        status[0].push_back(0);
        status[1].push_back(0);
        status[2].push_back(0);
        status[3].push_back(0);
    }
    std::vector<int> V[4];
    // std::set<int> V[2];
    int v_from;
    double value;
    double add;

    start = clock();
    for (int batch = 0; batch < 1; batch++)
    {
        int temp = rand() % G.n_vertex;
        // std::cout<<temp<<" ";
        V[0].clear();
        V[1].clear();
        V[2].clear();
        V[3].clear();
        V[0].push_back(temp);
        // V[0].insert(batch);
        status[0][temp] = 1;
        for (int step = 0; step < 2; step++)
        {
            for (auto it = V[step].begin(); it != V[step].end(); it++)
            {
                value = status[step][*it];
                status[step][*it] = 0;
                v_from = *it;

                for (int e_type = 0; e_type < G.n_edge_type; e_type++)
                {
                    add = value * R[e_type];
                    for (auto ptrv_to = G.linklist[v_from][e_type].begin(); ptrv_to != G.linklist[v_from][e_type].end(); ptrv_to++)
                    {
                        if (status[step+1][*ptrv_to] == 0)
                            V[step+1].push_back(*ptrv_to);
                        status[step+1][*ptrv_to] += add;
                        // V[(step+1)%2].insert(*ptrv_to);
                    }
                }
            }
            // std::cout<<V[step + 1].size()<<std::endl;
        }
        for (int e_type = 0; e_type < G.n_edge_type; e_type++)
        {
            for (auto ptrv = G.linklist[0][e_type].begin(); ptrv !=G.linklist[0][e_type].end(); ptrv++)
            {
                status[3][0] += status[2][*ptrv];
            }
        }
        // for (auto it = V[3].begin(); it != V[3].end(); it++)
        // {
        //     status[3][*it] = 0;
        // }

    }
    end = clock();
	double endtime=(double)(end-start)/CLOCKS_PER_SEC;
    std::cout<<endtime<<std::endl;

    return 0;
}

int test1()
{
    clock_t start,end;
    srand(time(0));
    int v_from, v_to;

    KnowledgeGraph KG;
    KG.load_data((char*)"../data/umls/train.txt");
    // KG.load_data((char*)"../data/wn18rr/train.txt");
    // KG.load_data((char*)"../data/FB15k-237/train.txt");
    Graph G(KG.n_entity, KG.n_relation_with_reverse());
    for (auto it = KG.triplets.begin(); it != KG.triplets.end(); it++)
    {
        G.add_edge(it->h, it->t, it->r);
    }
    v_from = rand() % G.n_vertex;
    v_to = rand() % G.n_vertex;
    dRule x(3, &G);
    
    x.forward_R();

    start = clock();
    x.forward(0, 1);
    x.backward(1.0);
    x.update();
    end = clock();
	double endtime=(double)(end-start)/CLOCKS_PER_SEC;
    std::cout<<"Time: "<<endtime<<std::endl;

    auto ved = vector<int>();
    ved.push_back(1);
    cout<<x.forward(0, ved)[0]<<endl<<endl;
    // for (int i = 0; i < G.n_edge_type; i++)
    // {
    //     std::cout<<x.params[0][i].var.value<<" ";
    // }
    // std::cout<<std::endl<<endl;
    // double sum = 0;
    // for (int i = 0; i < G.n_edge_type; i++)
    // {
    //     std::cout<<x.R[0][i].value<<" ";
    //     sum += x.R[0][i].value;
    // }
    // std::cout<<std::endl<<sum<<std::endl;

    return 0;
}