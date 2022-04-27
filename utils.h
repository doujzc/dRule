#pragma once
#include <vector>
#include <random>
#include <map>
#include <algorithm>
#include <cstring>
#include <queue>
#include <iostream>
#include <fstream>

typedef float real_t;

static std::default_random_engine _gen;
static std::normal_distribution<real_t> _norm(0, 1);
inline real_t randn()
{
    return _norm(_gen);
}

struct Triplet
{
    int h, r, t;
};


struct Variable
{
    real_t value, grad;

    Variable();
};

struct Parameter
{
    Variable var;
    real_t m, v, vm, t, total_grad;
    
    Parameter(bool randinit=true);

    void clear();
    void update(real_t lr=1e-3);
};

struct Rule
{
    std::vector<int> r_body;
    int head;
    Parameter weight;
    void clear();
};

class Graph
{
public:
    Graph(int _n_vertex, int _n_edge_type);
    ~Graph();
    
    int n_vertex;
    int n_edge_type;
    int offset;
    // linklist[v_from][e] = neibor of v_from via e.
    std::vector<int> **linklist;

    inline int rev(int edge_id)
    {
        return edge_id % 2 ? edge_id - 1 : edge_id + 1;
    }

    void add_edge(int v_from, int v_to, int edge_type);
};