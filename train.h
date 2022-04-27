#include <vector>
#include <iostream>

#include "utils.h"
#include "rule.h"


class RuleTrainer
{
public:

    Graph *G;
    dRule *R;

    // vto[e][v_from] = neibor of v_from via e.
    std::vector<int> **vto;
    std::vector<real_t> **weight;
    int **npos;
    int *total_pos;
    int *total_neg;
    std::vector<int> *vfrom;

    std::vector<int> dest;
    int *rec;

    RuleTrainer(int max_step, Graph* _G);
    ~RuleTrainer();

    void rule_dest(Rule *r, int vst);
    std::pair<int, int> accuracy(Rule *r);
    bool reweight(Rule *r);
    void negative_sample_all();
    void negative_sample(real_t rate=3.0);

    // compute l0, l1.
    std::pair<real_t, real_t> l_value(int head_type);
    Rule train_one_rule(int head_type, int n_epoch);
    void train(int n_rule, int n_epoch, std::vector<Rule> *rules);
};



