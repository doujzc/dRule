#pragma once
#include <vector>
#include <map>
#include <math.h>
#include <string.h>
#include <queue>
#include <set>
#include <algorithm>

#include "utils.h"

#define MAXLEN (256)
#define MAXBODYLEN (8)

class KnowledgeGraph
{
public:
    KnowledgeGraph();
    ~KnowledgeGraph();

    int n_entity, n_relation;
    std::vector<Triplet> triplets;
    std::map<std::string, int> e2id, r2id;
    std::map<int, std::string> id2e, id2r;
    inline int n_relation_without_reverse() {
        return this->n_relation;
    }
    inline int n_relation_with_reverse() {
        return this->n_relation * 2;
    }

    void load_data(char *path);
};