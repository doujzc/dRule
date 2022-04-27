#include "train.h"


RuleTrainer::RuleTrainer(int max_step, Graph* _G)
{
    G = _G;
    R = new dRule(max_step, _G);
    vto = new std::vector<int> *[_G->n_edge_type];
    npos = new int *[_G->n_edge_type];
    weight = new std::vector<real_t> *[_G->n_edge_type];
    vfrom = new std::vector<int>[_G->n_edge_type];
    rec = new int[_G->n_vertex];
    total_pos = new int[_G->n_edge_type];
    total_neg = new int[_G->n_edge_type];
    for (int v = 0; v < _G->n_vertex; v++)
    {
        rec[v] = 0;
    }
    for (int k = 0; k < _G->n_edge_type; k++)
    {
        vto[k] = new std::vector<int>[_G->n_vertex];
        npos[k] = new int[_G->n_vertex];
        weight[k] = new std::vector<real_t>[_G->n_vertex];
        total_pos[k] = 0;
        total_neg[k] = 0;
    }

    for (int k = 0; k < _G->n_edge_type; k++)
    {
        for (int v_from = 0; v_from < _G->n_vertex; v_from++)
        {
            npos[k][v_from] = 0;
            if (G->linklist[v_from][k].size() > 0)
            {
                vfrom[k].push_back(v_from);
                npos[k][v_from] = G->linklist[v_from][k].size();
                total_pos[k] += npos[k][v_from];
            }
            for (auto ptrvto = G->linklist[v_from][k].begin(); ptrvto != G->linklist[v_from][k].end(); ptrvto++)
            {
                vto[k][v_from].push_back(*ptrvto);
                weight[k][v_from].push_back(1.0);
            }
        }
    }
}

RuleTrainer::~RuleTrainer()
{
    for (int k = 0; k < G->n_edge_type; k++)
    {
        delete[] vto[k];
        delete[] npos[k];
        delete[] weight[k];
    }
    delete[] vto;
    delete[] npos;
    delete[] weight;
    delete[] vfrom;
    delete R;
    delete[] rec;
}

void RuleTrainer::rule_dest(Rule *r, int vst)
{
    for (auto ptrv = dest.begin(); ptrv != dest.end(); ptrv++)
    {
        rec[*ptrv] = 0;
    }
    dest.clear();

    std::queue< std::pair<int, int> > queue;
    queue.push(std::make_pair(vst, 0));
    int current_e, current_d, current_r, next_e;
    while (!queue.empty())
    {
        std::pair<int, int> pair = queue.front();
        current_e = pair.first;
        if (current_e == -1) return;
        current_d = pair.second;
        queue.pop();
        if (current_d == int(r->r_body.size()))
        {
            if (rec[current_e] == 0)
            {
                dest.push_back(current_e);
                rec[current_e] = 1;
            }
            continue;
        }
        current_r = r->r_body[current_d];
        for (auto ptrvto = G->linklist[current_e][current_r].begin(); ptrvto != G->linklist[current_e][current_r].end(); ptrvto++)
        {
            next_e = *ptrvto;
            queue.push(std::make_pair(next_e, current_d + 1));
        }
    }


}

std::pair<int, int> RuleTrainer::accuracy(Rule *r)
{
    int n0 = 0, n1 = 0;
    // for (auto ptrvfrom = vfrom[r->head].begin(); ptrvfrom != vfrom[r->head].end(); ptrvfrom++)
    // {
    //     rule_dest(r, *ptrvfrom);
    //     for (int i = 0; i < npos[r->head][*ptrvfrom]; i++)
    //         n0 += rec[vto[r->head][*ptrvfrom][i]];
    //     n1 += dest.size();
    // }

    for (int vfrom = 0; vfrom < G->n_vertex; vfrom++)
    {
        rule_dest(r, vfrom);
        for (int i = 0; i < npos[r->head][vfrom]; i++)
            n0 += rec[vto[r->head][vfrom][i]];
        n1 += dest.size();
    }
    return std::pair<int, int>{n0, n1};
}

bool RuleTrainer::reweight(Rule *r)
{
    if (r == NULL)
    {
        for (int k = 0; k < G->n_edge_type; k++)
        {
            for (auto ptrvfrom = vfrom[k].begin(); ptrvfrom != vfrom[k].end(); ptrvfrom++)
            {
                for (int i = 0; i < npos[k][*ptrvfrom]; i++)
                {
                    weight[k][*ptrvfrom][i] = 1.0;
                }
                for (int i = npos[k][*ptrvfrom]; i < weight[k][*ptrvfrom].size(); i++)
                {
                    weight[k][*ptrvfrom][i] = (G->n_vertex * G->n_vertex - total_pos[k]) / total_neg[k];
                }
            }
        }

        return true;
    }
    // return;
    std::pair<int, int> accpair = accuracy(r);
    if (accpair.second == 0) return true;
    real_t acc = (real_t)accpair.first / (real_t)accpair.second;
    real_t sumpospre = 0, sumnegpre = 0, sumposcur = 0, sumnegcur = 0;
    for (auto ptrvfrom = vfrom[r->head].begin(); ptrvfrom != vfrom[r->head].end(); ptrvfrom++)
    {
        rule_dest(r, *ptrvfrom);
        for (int i = 0; i < npos[r->head][*ptrvfrom]; i++)
        {
            sumpospre += weight[r->head][*ptrvfrom][i];
            weight[r->head][*ptrvfrom][i] *= 1 - acc * rec[vto[r->head][*ptrvfrom][i]];
            sumposcur += weight[r->head][*ptrvfrom][i];
        }
        for (int i = npos[r->head][*ptrvfrom]; i < weight[r->head][*ptrvfrom].size(); i++)
        {
            sumnegpre += weight[r->head][*ptrvfrom][i];
            weight[r->head][*ptrvfrom][i] /= 1 - acc * rec[vto[r->head][*ptrvfrom][i]] + 1e-10;
            sumnegcur += weight[r->head][*ptrvfrom][i];
        }
    }

    sumposcur = sumpospre / (sumposcur + 1e-5);
    sumnegcur = sumnegpre / (sumnegcur + 1e-5);

    for (auto ptrvfrom = vfrom[r->head].begin(); ptrvfrom != vfrom[r->head].end(); ptrvfrom++)
    {
        for (int i = 0; i < npos[r->head][*ptrvfrom]; i++)
        {
            weight[r->head][*ptrvfrom][i] *= sumposcur;
        }
        for (int i = npos[r->head][*ptrvfrom]; i < weight[r->head][*ptrvfrom].size(); i++)
        {
            weight[r->head][*ptrvfrom][i] *= sumnegcur;
        }
    }
    return true;
}

void RuleTrainer::negative_sample_all()
{
    // int *a = new int[G->n_vertex];

    // for (int k = 0; k < G->n_edge_type; k++)
    // {
    //     vfrom[k].clear();
    //     for (int v = 0; v < G->n_vertex; v++)
    //         vfrom[k].push_back(v);

    //     for (int vst = 0; vst < G->n_vertex; vst++)
    //     {
    //         std::fill(a, a + G->n_vertex, 0);
    //         for (auto ptrvto = vto[k][vst].begin(); ptrvto != vto[k][vst].end(); ptrvto++)
    //             a[*ptrvto] = 1;
    //         for (int ved = 0; ved < G->n_vertex; ved++)
    //         {
    //             if (a[ved] == 0)
    //             {
    //                 vto[k][vst].push_back(ved);
    //                 weight[k][vst].push_back(1.0);
    //                 total_neg[k]++;
    //             }
    //         }
    //     }
    // }
    // delete[] a;
    // return;

    // for (int k = 0; k < G->n_edge_type; k++)
    // {
    //     vfrom[k].clear();
    //     for (int v = 0; v < G->n_vertex; v++)
    //         vfrom[k].push_back(v);
    // }
    int a[G->n_edge_type][G->n_vertex][G->n_vertex];
    for (int k = 0; k < G->n_edge_type; k++)
    {
        for (int i = 0; i < G->n_vertex; i++)
        {
            for (int j = 0; j < G->n_vertex; j++)
            {
                // if (npos[k][i] > 0 && (real_t)(rand()) / (real_t)RAND_MAX < 3 * (real_t)total_pos[k] / (real_t)(G->n_vertex * vfrom[k].size()))
                if (false)
                {
                    a[k][i][j] = 1;
                }
                else a[k][i][j] = 0;
            }
        }
    }

    for (int k = 0; k < G->n_edge_type; k++)
    {
        for (int i = 0; i < vfrom[k].size(); i++)
        {
            for (int j = 0; j < 3 * vto[k][vfrom[k][i]].size(); j++)
            {
                a[k][vfrom[k][i]][rand() % G->n_vertex] = 1;
            }
        }
    }

    for (int k = 0; k < G->n_edge_type; k++)
    {
        for (int i = 0; i < G->n_vertex; i++)
        {
            for (auto ptrvto = G->linklist[i][k].begin(); ptrvto != G->linklist[i][k].end(); ptrvto++)
            {
                a[k][i][*ptrvto] = 0;
            }
        }
    }
    int sum = 0;
    real_t w;
    int rec = 0;

    for (int k = 0; k < G->n_edge_type; k++)
    {
        rec += total_pos[k];
        for (int i = 0; i < G->n_vertex; i++)
        {
            for (int j = 0; j < G->n_vertex; j++)
            {
                sum += a[k][i][j];
            }
        }
    }
    // w = (real_t)(G->n_vertex * G->n_vertex * G->n_edge_type - rec) / (real_t)sum;
// debug
// std::cout<<"sum: "<<sum<<"\n";
// std::cout<<"w: "<<w<<"\n";

    for (int k = 0; k < G->n_edge_type; k++)
    {
        sum = 0;
        for (int i = 0; i < G->n_vertex; i++)
            for (int j = 0; j < G->n_vertex; j++)
            {
                sum += a[k][i][j];
                total_neg[k] += a[k][i][j];
            }
        w = (real_t)(G->n_vertex * G->n_vertex - total_pos[k]) / (real_t)sum;
// debug
// w = 1.0;
        for (int i = 0; i < G->n_vertex; i++)
        {
            for (int j = 0; j < G->n_vertex; j++)
            {
                if (a[k][i][j])
                {
                    vto[k][i].push_back(j);
                    weight[k][i].push_back(w);
                }
            }
        }
    }
}

void RuleTrainer::negative_sample(real_t rate)
{
    if (rate == 0)
    {
        negative_sample_all();
        return;
    }

    int randvto = 0;
    for (int k = 0; k < G->n_edge_type; k++)
    {
        for (auto ptrvfrom = vfrom[k].begin(); ptrvfrom != vfrom[k].end(); ptrvfrom++)
        {
            for (auto ptrvto = vto[k][*ptrvfrom].begin(); ptrvto != vto[k][*ptrvfrom].end(); ptrvto++)
            {
                rec[*ptrvto] = 1;
            }
            int num = rate * vto[k][*ptrvfrom].size();
            for (int cnt = 0; cnt < num; cnt++)
            {
                randvto = rand() % G->n_vertex;
                if (rec[randvto] == 1) continue;
                rec[randvto] = 1;
                vto[k][*ptrvfrom].push_back(randvto);
                weight[k][*ptrvfrom].push_back(-1);
                total_neg[k]++;
            }
            for (auto ptrvto = vto[k][*ptrvfrom].begin(); ptrvto != vto[k][*ptrvfrom].end(); ptrvto++)
            {
                rec[*ptrvto] = 0;
            }
        }
    }
    int sum = 0;
    real_t w;
    for (int k = 0; k < G->n_edge_type; k++)
    {
        w = (real_t)(G->n_vertex * G->n_vertex - total_pos[k]) / (real_t)total_neg[k];
        for (auto ptrvfrom = vfrom[k].begin(); ptrvfrom != vfrom[k].end(); ptrvfrom++)
        {
            for (int i = npos[k][*ptrvfrom]; i < weight[k][*ptrvfrom].size(); i++)
            {
                weight[k][*ptrvfrom][i] = w;
            }
        }
    }
}

std::pair<real_t, real_t> RuleTrainer::l_value(int head_type)
{
    real_t l0 = 0.0, l1 = 0.0;
    std::vector<real_t> val;
    for (auto ptrvfrom = vfrom[head_type].begin(); ptrvfrom != vfrom[head_type].end(); ptrvfrom++)
    {
        val = R->forward(*ptrvfrom, vto[head_type][*ptrvfrom]);
        for (int i = 0; i < npos[head_type][*ptrvfrom]; i++)
        {
            l0 += weight[head_type][*ptrvfrom][i] * val[i];
        }
        for (int i = npos[head_type][*ptrvfrom]; i < vto[head_type][*ptrvfrom].size(); i++)
        {
            l1 += weight[head_type][*ptrvfrom][i] * val[i];
        }
    }
// debug
// std::cout<<"l: "<<l0<<" "<<l1<<"\n";

    return std::pair<real_t, real_t>{l0, l0 + l1};
}

Rule RuleTrainer::train_one_rule(int head_type, int n_epoch)
{
    R->reset();
    R->set_power(2);
    std::pair<real_t, real_t> l;
    real_t dl0, dl1;

    for (int epoch = 0; epoch < n_epoch; epoch++)
    {
        R->forward_R();
        l = l_value(head_type);
        if (l.first == 0) break;
        dl0 = -1 / l.second - 1.0 / ((real_t)total_pos[head_type] + 1.0);
        // dl0 = std::isinf(dl0) ? 1e20 : dl0;
        dl1 = l.first / (l.second * l.second);
        // dl1 = std::isinf(dl1) ? 1e20 : dl1;
        dl0 += dl1;
// debug----------------------------
// std::cout<<"pos: "<<weight[head_type][0][0] * dl0<<"\n";
// std::cout<<"neg: "<<weight[head_type][0][npos[head_type][0]] * dl1<<"\n";
// std::cout<<"loss: "<< -(real_t)l.first/(real_t)l.second-(real_t)l.first/((real_t)total_pos[head_type] + 1.0)<<"\n";
// std::cout<<"l0: "<<l.first<<" l1: "<<l.second<<"\n";
// std::cout<<"dl0: "<<dl0 - dl1<<" dl1: "<<dl1<<"\n";
// real_t cnt=0;
// for (int i = 0; i < G->n_vertex; i++)
// for (int j = 0; j < vto[head_type][i].size(); j++)
// {
//     cnt += weight[head_type][i][j];
// }
// std::cout<<"total weight: "<<cnt<<"\n";
// std::cout<<"-----------\n";
// ----------------------------------
        std::vector<real_t> grad;
        for (auto ptrvfrom = vfrom[head_type].begin(); ptrvfrom != vfrom[head_type].end(); ptrvfrom++)
        {
            R->forward(*ptrvfrom, vto[head_type][*ptrvfrom]);
// debug---------------
// std::ofstream fout("log.txt", std::ios::app);
// fout<<"\n----------------------------\n";
// fout<<"Vfrom: "<<*ptrvfrom<<"\n";
// fout<<"Vto:\n";
// for (int i = 0; i < G->n_vertex; i++)
// {
//     fout<<R->V[R->n_step][i].value<<" ";
// }
// fout<<"\n----------------------------\n";
// fout.close();
// ---------------------
            grad.clear();

            for (int i = 0; i < npos[head_type][*ptrvfrom]; i++)
                grad.push_back(weight[head_type][*ptrvfrom][i] * dl0);
            for (int i = npos[head_type][*ptrvfrom]; i < vto[head_type][*ptrvfrom].size(); i++)
                grad.push_back(weight[head_type][*ptrvfrom][i] * dl1);

                // grad.push_back(0.3);


            R->backward(grad);

        }

        R->backward_R();

// debug----------------------------------------------------------
// std::ofstream fout("log.txt", std::ios::app);
// fout<<"\n----------------------------\n";
// fout<<"Head Type: "<<head_type<<"\n";
// fout<<"Epoch: "<<epoch<<"\n";
// fout<<"l0, l1: "<<l.first<<" "<<l.second<<"\n";
// fout<<"grad:\n";
// for (int i = 0; i < R->n_step; i++)
// {
//     for (int j = 0; j < R->n_edge_type; j++)
//     {
// // std::cout<<R->params[i][j].total_grad<<" ";
// fout<<R->params[i][j].total_grad<<" ";
//     }
// // std::cout<<"\n";
// fout<<"\n";
// }
// // std::cout<<"\n";
// fout<<"\n";
// fout<<"V: Vfrom = "<<vfrom[head_type].back()<<"\n";
// for (int i = 0; i <= R->n_step; i++)
// {
//     for (int j = 0; j < G->n_vertex; j++)
//     {
//         fout<<"("<<R->V[i][j].value<<", "<<R->V[i][j].grad<<") ";
//     }
//     fout<<"\n";
// }
// fout<<"\n";
// fout<<"R:\n";
// for (int i = 0; i < R->n_step; i++)
// {
//     for (int j = 0; j < R->n_edge_type; j++)
//     {
// fout<<"("<<R->R[i][j].value<<", "<<R->R[i][j].grad<<") ";
//     }
// fout<<"\n";
// }
// fout<<"\n--------------------\n";
// fout.close();
// ----------------------------------------------------------------
        
        R->update();
    }
    // std::cout<<"\n-------------\n";
    Rule ret = R->to_rule();
    ret.head = head_type;
    return ret;
}

void RuleTrainer::train(int n_rule, int n_epoch, std::vector<Rule> *rules)
{
    std::vector<Rule> vec;
    Rule r;
    for (int k = 0; k < 3; k++)
    {
        vec.clear();
        for (int i = 0; i < n_rule; i++)
        {
            std::cout<<k<<" "<<i<<"\n";
            r = train_one_rule(k, n_epoch);
            if (!reweight(&r)) continue;
            vec.push_back(r);
        }
        rules[k] = vec;
    }
}
