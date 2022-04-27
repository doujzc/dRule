#include "kg.h"

KnowledgeGraph::KnowledgeGraph()
{
    n_entity = 0;
    n_relation = 0;
    
    e2id.clear(); r2id.clear();
    id2e.clear(); id2r.clear();
    triplets.clear();
}

KnowledgeGraph::~KnowledgeGraph()
{
}

void KnowledgeGraph::load_data(char *path)
{
    FILE *f;
    char hs[MAXLEN], rs[MAXLEN], ts[MAXLEN], rev[MAXLEN];
    int cur_eid = 0;
    int cur_rid = 0;
    int h, r, t, rev_r;
    f = fopen(path, "rb");
    if (f == NULL)
    {
        printf("ERROR: Open file failed.\n");
        exit(1);
    }

    while (true)
    {
        if (fscanf(f, "%s %s %s", hs, rs, ts) != 3) break;
        if (e2id.count(hs) == 0)
        {
            e2id[hs] = cur_eid;
            id2e[cur_eid] = hs;
            cur_eid++;
        }
        if (e2id.count(ts) == 0) 
        {
            e2id[ts] = cur_eid;
            id2e[cur_eid] = ts;
            cur_eid++;
        }
        if (r2id.count(rs) == 0)
        {
            r2id[rs] = cur_rid;
            id2r[cur_rid] = rs;
            cur_rid++;
            strcpy(rev, "rev-");
            strcat(rev, rs);
            r2id[rev] = cur_rid;
            id2r[cur_rid] = rev;
            cur_rid++;
        }
        
        h = e2id[hs];
        r = r2id[rs];
        rev_r = r + 1;
        t = e2id[ts];

        this->triplets.push_back({h, r, t});
        this->triplets.push_back({t, rev_r, h});
    }
    fclose(f);

    this->n_entity = cur_eid;
    this->n_relation = cur_rid / 2;

    printf("#Entities: %d\n", n_entity);
    printf("#Relations with Reverse: %d\n", n_relation_with_reverse());
    printf("#Triplets: %ld\n", triplets.size());
}