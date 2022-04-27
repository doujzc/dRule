#include "rule.h"

dRule::dRule(int _n_step, Graph *_G, bool randinit)
{
    n_edge_type = _G->n_edge_type;
    n_step = _n_step;
    pow = 2.0;

    params = new Parameter *[n_step];
    R = new Variable *[n_step];
    V = new Variable *[n_step + 1];
    for (int l = 0; l < n_step; l++)
    {
        params[l] = new Parameter[n_edge_type];
        for (int k = 0; k < n_edge_type; k++)
        {
            params[l][k] = Parameter(randinit);
        }
        R[l] = new Variable[n_edge_type];
        V[l] = new Variable[_G->n_vertex];
    }
    V[n_step] = new Variable[_G->n_vertex];
    Vid = new std::vector<int>[n_step + 1];
    G = _G;
    _vst = -1;
    _ved.clear();
    visited = new bool[_G->n_vertex];
}

dRule::~dRule()
{
    for (int l = 0; l < n_step; l++)
    {
        delete[] params[l];
        delete[] R[l];
        delete[] V[l];
    }
    delete[] params;
    delete[] R;
    delete[] V;
    delete[] Vid;
    delete[] visited;
}


void dRule::set_power(real_t power)
{
    pow = power;
}

void dRule::init_V_Vid()
{
    for (int l = 0; l <= n_step; l++)
        Vid[l].clear();
    Vid[0].push_back(_vst);
    for (auto it = _ved.begin(); it != _ved.end(); it++)
    {
        Vid[n_step].push_back(*it);
        V[n_step][*it].value = 0.0;
        V[n_step][*it].grad = 0.0;
    }
    V[0][_vst].value = 1.0;
    V[0][_vst].grad = 0.0;
    
    for (int l = 1; l < n_step - 1; l++)
    {
        for (auto it = visited_id.begin(); it != visited_id.end(); it++)
            visited[*it] = 0;
        visited_id.clear();
        for (int k = 0; k < n_edge_type; k++)
        {
            for (auto ptr_vfrom = Vid[l - 1].begin(); ptr_vfrom != Vid[l - 1].end(); ptr_vfrom++)
            {
                for (auto ptr_vto = G->linklist[*ptr_vfrom][k].begin(); ptr_vto != G->linklist[*ptr_vfrom][k].end(); ptr_vto++)
                {
                    if (!visited[*ptr_vto])
                    {
                        visited[*ptr_vto] = true;
                        visited_id.push_back(*ptr_vto);
                        Vid[l].push_back(*ptr_vto);
                        V[l][*ptr_vto].value = 0.0;
                        V[l][*ptr_vto].grad = 0.0;
                    }
                }
            }
        }
    }
    for (auto it = visited_id.begin(); it != visited_id.end(); it++)
        visited[*it] = 0;
    visited_id.clear();
    if (n_step >= 2)
    {
        for (auto ptr_ved = _ved.begin(); ptr_ved != _ved.end(); ptr_ved++)
        {
            for (int k = 0; k < n_edge_type; k++)
            {
                for (auto ptr_vfrom = G->linklist[*ptr_ved][G->rev(k)].begin(); ptr_vfrom != G->linklist[*ptr_ved][G->rev(k)].end(); ptr_vfrom++)
                {
                    if (!visited[*ptr_vfrom])
                    {
                        visited[*ptr_vfrom] = true;
                        visited_id.push_back(*ptr_vfrom);
                        Vid[n_step - 1].push_back(*ptr_vfrom);
                        V[n_step - 1][*ptr_vfrom].value = 0.0;
                        V[n_step - 1][*ptr_vfrom].grad = 0.0;
                    }
                }
            }
        }
    }
}

void dRule::forward_R()
{
    real_t sum;
    real_t e[n_edge_type];

    for (int l = 0; l < n_step; l++)
    {
        sum = 0;
        for (int k = 0; k < n_edge_type; k++)
        {
            e[k] = exp(params[l][k].var.value);
            sum += e[k];
        }
        for (int k = 0; k < n_edge_type; k++)
        {
            R[l][k].value = std::pow(e[k] / sum, pow);
            R[l][k].grad = 0;
        }
    }
}

void dRule::backward_R()
{
// std::cout<<"\n-------\n";
    for (int l = 0; l < n_step; l++)
    {
        for (int j = 0; j < n_edge_type; j++)
        {
            R[l][j].value = std::pow(R[l][j].value, 1 / pow);
            R[l][j].grad *= pow * std::pow(R[l][j].value, pow - 1);
// std::cout<<R[l][j].grad<<" ";
        }
// std::cout<<"\n";
    }
// std::cout<<"\n-------\n";
    for (int l = 0; l < n_step; l++)
    {
        for (int j = 0; j < n_edge_type; j++)
        {
//debug
// std::cout<<"\n"<<params[l][j].total_grad<<"\n";
            params[l][j].total_grad += R[l][j].grad * R[l][j].value;
//debug
// std::cout<<R[l][j].grad<<" "<<R[l][j].value<<"\n";
            for (int i = 0; i < n_edge_type; i++)
            {
// debug
// std::cout<<params[l][j].total_grad<<"\n";
                params[l][j].total_grad -= R[l][i].grad * R[l][i].value * R[l][j].value;
            }
        }
    }
}

void dRule::f1(int step)
{
    double add;
    int vf, vt;
    int size0, size1;
    int revk;
    real_t r;

    size0 = Vid[step - 1].size();
    for (int i = 0; i < size0; i++)
    {
        vf = Vid[step - 1][i];
        for (int k = 0; k < n_edge_type; k++)
        {
            add = V[step - 1][vf].value * R[step - 1][k].value;
            size1 = G->linklist[vf][k].size();
            for (int j = 0; j < size1; j++)
            {
                V[step][G->linklist[vf][k][j]].value += add;
            }
        }
    }
}

void dRule::f2(int step)
{
    double add;
    int vf, vt;
    int size0, size1;
    int revk;
    real_t r;

    size0 = Vid[step].size();
    for (int i = 0; i < size0; i++)
    {
        vt = Vid[step][i];
        for (int k = 0; k < n_edge_type; k++)
        {
            revk = G->rev(k);
            size1 = G->linklist[vt][revk].size();
            for (int j = 0; j < size1; j++)
            {
                V[step][vt].value += V[step - 1][G->linklist[vt][revk][j]].value * R[step - 1][k].value;
            }
        }
    }
}

void dRule::forward_step(int step)
{
    double add;
    int vf, vt;
    int size0, size1;
    int revk;
    real_t r;

    if (Vid[step - 1].size() < Vid[step].size())
    {
        f1(step);
    }
    else
    {
        f2(step);
    }

    // if (Vid[step - 1].size() < Vid[step].size())
    // {
    //     for (auto ptr_vfrom = Vid[step - 1].begin(); ptr_vfrom != Vid[step - 1].end(); ptr_vfrom++)
    //     {
    //         for (int k = 0; k < n_edge_type; k++)
    //         {
    //             add = V[step - 1][*ptr_vfrom].value * R[step - 1][k].value;
    //             for (auto ptr_vto = G->linklist[*ptr_vfrom][k].begin(); ptr_vto != G->linklist[*ptr_vfrom][k].end(); ptr_vto++)
    //             {
    //                 V[step][*ptr_vto].value += add;
    //             }
    //         }
    //     }
    // }
    // else
    // {
    //     for (auto ptr_vto = Vid[step].begin(); ptr_vto != Vid[step].end(); ptr_vto++)
    //     {
    //         for (int k = 0; k < n_edge_type; k++)
    //         {
    //             for (auto ptr_vfrom = G->linklist[*ptr_vto][G->rev(k)].begin(); ptr_vfrom != G->linklist[*ptr_vto][G->rev(k)].end(); ptr_vfrom++)
    //             {
    //                 V[step][*ptr_vto].value += V[step - 1][*ptr_vfrom].value * R[step - 1][k].value;
    //             }
    //         }
    //     }
    // }

    for (auto ptr_vto = Vid[step].begin(); ptr_vto != Vid[step].end(); ptr_vto++)
    {
        V[step][*ptr_vto].value = V[step][*ptr_vto].value < 1.0 ? V[step][*ptr_vto].value : 1.0;
    }
}

void dRule::backward_step(int step)
{
    real_t add;

    if (Vid[step - 1].size() < Vid[step].size())
    {
        for (auto ptr_vfrom = Vid[step - 1].begin(); ptr_vfrom != Vid[step - 1].end(); ptr_vfrom++)
        {
            for (int k = 0; k < n_edge_type; k++)
            {
                for (auto ptr_vto = G->linklist[*ptr_vfrom][k].begin(); ptr_vto != G->linklist[*ptr_vfrom][k].end(); ptr_vto++)
                {
                    V[step - 1][*ptr_vfrom].grad += V[step][*ptr_vto].grad * R[step - 1][k].value;
                    R[step - 1][k].grad += V[step - 1][*ptr_vfrom].value * V[step][*ptr_vto].grad;
                }
            }
        }
    }
    else
    {
        for (auto ptr_vto = Vid[step].begin(); ptr_vto != Vid[step].end(); ptr_vto++)
        {
            for (int k = 0; k < n_edge_type; k++)
            {
                for (auto ptr_vfrom = G->linklist[*ptr_vto][G->rev(k)].begin(); ptr_vfrom != G->linklist[*ptr_vto][G->rev(k)].end(); ptr_vfrom++)
                {
                    V[step - 1][*ptr_vfrom].grad += V[step][*ptr_vto].grad * R[step - 1][k].value;
                    R[step - 1][k].grad += V[step - 1][*ptr_vfrom].value * V[step][*ptr_vto].grad;
                }
            }
        }
    }

    // if (step == 3)
    // {
    // for (int i = 0; i < G->n_vertex; i++)
    // {
    //     std::cout<<V[step][i].grad<<" ";
    // }
    // std::cout<<"\n\n";
    // for (int i = 0; i < G->n_edge_type; i++)
    // {
    //     std::cout<<R[step-1][i].grad<<" ";
    // }

    // std::cout<<"\n-------------\n";
    // }

}

std::vector<real_t> dRule::forward(int vst, std::vector<int> ved)
{
    _vst = vst;
    _ved = ved;
    init_V_Vid();

    for (int l = 1; l <= n_step; l++)
    {
        forward_step(l);
    }
    std::vector<real_t> ret;
    for (auto it = ved.begin(); it != ved.end(); it++)
    {
        ret.push_back(V[n_step][*it].value);
    }
    return ret;
}

void dRule::backward(std::vector<real_t> grad)
{
    auto ptr_v = _ved.begin();
    for (auto ptr_grad = grad.begin(); ptr_grad != grad.end(); ptr_grad++)
    {
        V[n_step][*ptr_v].grad = *ptr_grad;
        ptr_v++;
    }
    for (int l = n_step; l > 0; l--)
    {
        backward_step(l);
    }
}

real_t dRule::forward(int vst, int ved)
{
    std::vector<int> vec_ved;
    vec_ved.push_back(ved);
    return forward(vst, vec_ved)[0];
}

void dRule::backward(real_t grad)
{
    std::vector<real_t> vec_grad;
    vec_grad.push_back(grad);
    backward(vec_grad);
}

void dRule::update()
{
    real_t lr = 1e-2;
    real_t m = 0;
    real_t rec = 1;
    for (int l = 0; l < n_step; l++)
    {
        m = 0;
        for (int k = 0; k < n_edge_type; k++)
        {
            m = m > R[l][k].value ? m : R[l][k].value;
        }
        rec *= m;
    }



    for (int l = 0; l < n_step; l++)
    {
        for (int k = 0; k < n_edge_type; k++)
        {
            params[l][k].update(lr / rec);
        }
    }

    for (int l = 0; l < n_step; l++)
    {
        m = params[l][0].var.value;
        for (int k = 0; k < n_edge_type; k++)
        {
            m = m > params[l][k].var.value ? m : params[l][k].var.value;
        }
        for (int k = 0; k < n_edge_type; k++)
        {
            params[l][k].var.value -= m;
        }
    }
}

void dRule::reset(bool randinit)
{
    for (int l = 0; l < n_step; l++)
    {
        for (int k = 0; k < n_edge_type; k++)
        {
            params[l][k] = Parameter(randinit);
        }
    }
    _ved.clear();
    _vst = -1;
}

Rule dRule::to_rule()
{
    forward_R();
    real_t m;
    int id;
    Rule ret = Rule();
    // std::cout<<"Rule Param: ";
    for (int l = 0; l < n_step; l++)
    {
        m = R[l][0].value;
        id = 0;
        for (int i = 1; i < n_edge_type; i++)
        {
            if (R[l][i].value > m) {
                m = R[l][i].value;
                id = i;
            }
        }
        ret.r_body.push_back(id);
        // std::cout<<m<<" ";
    }
    // std::cout<<std::endl;
    return ret;
}