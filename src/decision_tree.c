#include "decision_tree.h"
#include <stdlib.h>
#include <float.h>
#include <string.h>

typedef struct
{
    int feature_index;
    Sample *data;
} SortContext;

// Used for qsort_r to sort indices by feature value
static int cmp_by_feature(const void *a, const void *b, void *ctx)
{
    SortContext *sort_ctx = (SortContext *)ctx;
    int ia = *(int *)a;
    int ib = *(int *)b;
    double va = sort_ctx->data[ia].features[sort_ctx->feature_index];
    double vb = sort_ctx->data[ib].features[sort_ctx->feature_index];
    return (va > vb) - (va < vb);
}

// Computes impurity of a node using Gini index
double compute_gini(Sample *data, int *indices, int n, int n_classes)
{
    if (n == 0)
        return 0.0;

    int counts[10] = {0};
    for (int i = 0; i < n; i++)
        counts[data[indices[i]].label]++;

    // Gini = 1 - Σ(p_i^2)
    double gini = 1.0;
    for (int c = 0; c < n_classes; c++)
    {
        double p = (double)counts[c] / n;
        gini -= p * p;
    }

    return gini;
}

// Computes weighted Gini after a split (maybe inline?)
double gini_split(double gini_left, int n_left, double gini_right, int n_right)
{
    int total = n_left + n_right;
    return ((double)n_left / total) * gini_left + ((double)n_right / total) * gini_right;
}

// Finds class with most occurences in the given indices
int majority_class(Sample *data, int *indices, int n, int n_classes)
{
    int counts[10] = {0};
    for (int i = 0; i < n; i++)
        counts[data[indices[i]].label]++;
    int best = 0;
    for (int c = 1; c < n_classes; c++)
        if (counts[c] > counts[best])
            best = c;
    return best;
}

BestSplit find_best_split(Sample *data, int *indices, int n,
                          int n_features, int n_classes)
{
    BestSplit best = {-1, 0.0, DBL_MAX}; // shared best split across threads

#pragma omp parallel
    {
        // Each thread gets its own local arrays (stack allocated, private)
        int sorted[MAX_SAMPLES];
        int left_idx[MAX_SAMPLES];
        int right_idx[MAX_SAMPLES];

        // Each thread tracks its own local best
        BestSplit local_best = {-1, 0.0, DBL_MAX};

#pragma omp for schedule(dynamic) // many splits are trivial to evaluate (va == vb), so dynamic scheduling helps load balance
        for (int f = 0; f < n_features; f++)
        {
            SortContext sort_ctx = {f, data}; // context for sorting by this feature
            memcpy(sorted, indices, n * sizeof(int));
            qsort_r(sorted, n, sizeof(int), cmp_by_feature, &sort_ctx);

            // Test consecutive split points between unique feature values
            for (int i = 0; i < n - 1; i++)
            {
                double va = data[sorted[i]].features[f];
                double vb = data[sorted[i + 1]].features[f];
                if (va == vb) // trivial split, skip
                    continue;

                double threshold = (va + vb) / 2.0; 

                int nl = 0, nr = 0;
                for (int j = 0; j < n; j++)
                {
                    if (data[indices[j]].features[f] <= threshold)
                        left_idx[nl++] = indices[j];
                    else
                        right_idx[nr++] = indices[j];
                }

                // Compute impurity of each node
                double gl = compute_gini(data, left_idx, nl, n_classes);
                double gr = compute_gini(data, right_idx, nr, n_classes);
                double g = gini_split(gl, nl, gr, nr);

                if (g < local_best.gini ||
                    (g == local_best.gini && f < local_best.feature_index))
                 {
                     local_best.feature_index = f;
                     local_best.threshold     = threshold;
                     local_best.gini          = g;
                 }
            }
        }

// Merge each thread's local best into the global best
#pragma omp critical
        {
            if (local_best.gini < best.gini ||
                (local_best.gini == best.gini && 
                 local_best.feature_index < best.feature_index)) // tie-breaker for deterministic results, not relying on thread scheduling
             {
                 best = local_best;
             }
        }
    }

    return best;
}

// Recursively builds the decision tree by finding the best split at each node
Node *build_tree(Sample *data, int *indices, int n,
                 int n_features, int n_classes,
                 int depth, int max_depth)
{

    Node *node = calloc(1, sizeof(Node));
    node->n_samples = n;
    node->impurity = compute_gini(data, indices, n, n_classes);

    // Stopping criteria, make leaf
    int pure = (node->impurity < 1e-9);
    if (pure || depth >= max_depth || n <= 1)
    {
        node->is_leaf = 1;
        node->predicted_class = majority_class(data, indices, n, n_classes);
        return node;
    }

    // Split
    BestSplit split = find_best_split(data, indices, n, n_features, n_classes);

    // If no split was found, make leaf
    if (split.feature_index < 0)
    {
        node->is_leaf = 1;
        node->predicted_class = majority_class(data, indices, n, n_classes);
        return node;
    }

    node->feature_index = split.feature_index;
    node->threshold = split.threshold;

    // Split indices
    int left_idx[MAX_SAMPLES], right_idx[MAX_SAMPLES];
    int nl = 0, nr = 0;
    for (int i = 0; i < n; i++)
    {
        if (data[indices[i]].features[split.feature_index] <= split.threshold)
            left_idx[nl++] = indices[i];
        else
            right_idx[nr++] = indices[i];
    }

    node->left = build_tree(data, left_idx, nl, n_features, n_classes, depth + 1, max_depth);
    node->right = build_tree(data, right_idx, nr, n_features, n_classes, depth + 1, max_depth);

    return node;
}

// Recursively predicts class for given features by traversing the tree
int predict(Node *node, double *features)
{
    if (node->is_leaf)
        return node->predicted_class;
    if (features[node->feature_index] <= node->threshold)
        return predict(node->left, features);
    else
        return predict(node->right, features);
}

void free_tree(Node *node)
{
    if (!node)
        return;
    free_tree(node->left);
    free_tree(node->right);
    free(node);
}