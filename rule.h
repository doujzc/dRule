#pragma once
#include "utils.h"
static int cnt0 = 0;
static int cnt1 = 0;
static int cnt2 = 0;
class dRule
{
public:
    int n_edge_type;
    int n_step;
    real_t pow;
    Parameter **params;
    Variable **R;
    Variable **V;
    std::vector<int> *Vid;
    Graph* G;
    int _vst;
    std::vector<int> _ved;
    bool *visited;
    std::vector<int> visited_id;

    dRule(int _n_step, Graph *_G, bool randinit=true);
    ~dRule();
    void set_power(real_t power);
    void init_V_Vid();

    // clear R.grad
    void forward_R();
    void backward_R();

    //lazy clear
    // step = 1, ..., n_step
    // (step - 1).value -> step.value
    void f1(int step);
    void f2(int step);
    void forward_step(int step);
    // (step - 1).grad <- step.grad
    void backward_step(int step);

    std::vector<real_t> forward(int vst, std::vector<int> ved);
    void backward(std::vector<real_t> grad);

    real_t forward(int vst, int ved);
    void backward(real_t grad);

    void update();
    void reset(bool randinit=true);

    Rule to_rule();
};


