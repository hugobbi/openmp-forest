#include "decision_tree.h"
#include <stdlib.h>
#include <float.h>
#include <string.h>

// Used for qsort when finding best split
// TODO: this needs to be changed for parallel version
static int   g_sort_feature;
static Sample *g_sort_data;

// Computes impurity of a node using Gini index
double compute_gini(Sample *data, int *indices, int n, int n_classes) 
{
    if (n == 0) return 0.0;

    int counts[10] = {0};
    for (int i = 0; i < n; i++)
        counts[data[indices[i]].label]++;

    // Gini = 1 - Σ(p_i^2)
    double gini = 1.0;
    for (int c = 0; c < n_classes; c++) {  
        double p = (double)counts[c] / n;
        gini -= p * p;
    }

    return gini;
}

// Computes weighted Gini after a split (maybe inline?)
double gini_split(double gini_left,  int n_left, double gini_right, int n_right) 
{
    int total = n_left + n_right;
    return ((double)n_left  / total) * gini_left
         + ((double)n_right / total) * gini_right;    
}

// Finds class with most occurences in the given indices
int majority_class(Sample *data, int *indices, int n, int n_classes) 
{
    int counts[10] = {0};
    for (int i = 0; i < n; i++)
        counts[data[indices[i]].label]++;
    int best = 0;
    for (int c = 1; c < n_classes; c++)
        if (counts[c] > counts[best]) best = c;
    return best;
}

// Used for qsort to sort indices by feature value
int cmp_by_feature(const void *a, const void *b) 
{
    int ia = *(int*)a, ib = *(int*)b;
    double va = g_sort_data[ia].features[g_sort_feature];
    double vb = g_sort_data[ib].features[g_sort_feature];
    return (va > vb) - (va < vb);
}

// Finds the best split by testing all features and thresholds
BestSplit find_best_split(Sample *data, int *indices, int n, 
                            int n_features, int n_classes) 
{
    BestSplit best = { -1, 0.0, DBL_MAX };

    int left_idx[MAX_SAMPLES], right_idx[MAX_SAMPLES];

    for (int f = 0; f < n_features; f++) {
        // Sort indices by feature f
        int sorted[MAX_SAMPLES];
        memcpy(sorted, indices, n * sizeof(int));
        // Calls stdlib qsort, which will call cmp_by_feature
        g_sort_feature = f;
        g_sort_data = data;
        qsort(sorted, n, sizeof(int), cmp_by_feature);

        // Tests each split point between consecutive values
        for (int i = 0; i < n - 1; i++) {
            double va = data[sorted[i]].features[f];
            double vb = data[sorted[i + 1]].features[f];

            if (va == vb) continue; // values must be different

            double threshold = (va + vb) / 2.0;

            // Left <= threshold, Right > threshold
            int nl = 0, nr = 0;
            for (int j = 0; j < n; j++) {
            if (data[indices[j]].features[f] <= threshold)
                left_idx[nl++]  = indices[j];
            else
                right_idx[nr++] = indices[j];
            }

            double gl = compute_gini(data, left_idx,  nl, n_classes);
            double gr = compute_gini(data, right_idx, nr, n_classes);
            double g  = gini_split(gl, nl, gr, nr);

            if (g < best.gini) {
                best.feature_index = f;
                best.threshold = threshold;
                best.gini = g;
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
    if (pure || depth >= max_depth || n <= 1) {
        node->is_leaf = 1;
        node->predicted_class = majority_class(data, indices, n, n_classes);
        return node;
    }
    
    // Split
    BestSplit split = find_best_split(data, indices, n, n_features, n_classes);

    // If no split was found, make leaf
    if (split.feature_index < 0) {
        node->is_leaf = 1;
        node->predicted_class = majority_class(data, indices, n, n_classes);
        return node;
    }

    node->feature_index = split.feature_index;
    node->threshold = split.threshold;

    // Split indices
    int left_idx[MAX_SAMPLES], right_idx[MAX_SAMPLES];
    int nl = 0, nr = 0;
    for (int i = 0; i < n; i++) {
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
    if (node->is_leaf) return node->predicted_class;
    if (features[node->feature_index] <= node->threshold)
        return predict(node->left, features);
    else
        return predict(node->right, features);
}

void free_tree(Node *node) 
{
    if (!node) return;
    free_tree(node->left);
    free_tree(node->right);
    free(node);
}