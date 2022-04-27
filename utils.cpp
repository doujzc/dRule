#include "utils.h"


Parameter::Parameter(bool randinit)
{
    var.value = randinit ? randn() : 0.0;
    // var.value = 0;
    var.grad = 0;
    m = 0;
    v = 0;
    vm = 0;
    t = 0;
    total_grad = 0;
}

void Parameter::clear()
{
    m = 0;
    v = 0;
    vm = 0;
    t = 0;
    total_grad = 0;
    var.grad=0;
}

void Parameter::update(real_t lr)
{
    t += 1;
    m = 0.9 * m + 0.1 * total_grad;
    v = 0.999 * v + 0.001 * total_grad * total_grad;

    real_t bias1 = 1 - std::pow(0.9, t);
    real_t bias2 = 1 - std::pow(0.999, t);

    real_t mt = m / bias1;
    real_t vt = sqrt(v) / sqrt(bias2) + 0.00000001;

    var.value -= lr * mt / vt;
    total_grad = 0;
}

Variable::Variable()
{
    value = 0;
    grad = 0;
}

void Rule::clear()
{
    r_body.clear();
    head = -1;
    weight.clear();
}

Graph::Graph(int _n_vertex, int _n_edge_type)
{
    n_vertex = _n_vertex;
    n_edge_type = _n_edge_type;
    linklist = new std::vector<int> *[n_vertex];
    for (int i = 0; i < n_vertex; i++)
    {
        linklist[i] = new std::vector<int>[n_edge_type];
    }
    offset = _n_edge_type / 2;
}

Graph::~Graph()
{
    for (int i = 0; i < n_vertex; i++)
    {
        delete[] linklist[i];
    }
    delete[] linklist;
}

void Graph::add_edge(int v_from, int v_to, int edge_type)
{
    linklist[v_from][edge_type].push_back(v_to);
}