#define MAX_SAMPLES 100
#define MAX_FEATURES 10
#define MAX_DEPTH 5

typedef struct Node {
    int    is_leaf;       
    int    feature_index;   // feature used for splitting
    double threshold;      
    int    predicted_class; // (only in leaf nodes)
    double impurity;        // node impurity (e.g., Gini)
    int    n_samples;      
    struct Node *left;  
    struct Node *right;
} Node;

typedef struct {
    double features[MAX_FEATURES];
    int    label;
} Sample;

typedef struct {
    int    feature_index;
    double threshold;
    double gini;
} BestSplit;

double compute_gini(Sample *data, int *indices, int n, int n_classes);
double gini_split(double gini_left,  int n_left, double gini_right, int n_right);
int majority_class(Sample *data, int *indices, int n, int n_classes);
int cmp_by_feature(const void *a, const void *b);
BestSplit find_best_split(Sample *data, int *indices, int n, 
                          int n_features, int n_classes);
Node *build_tree(Sample *data, int *indices, int n,
                 int n_features, int n_classes,
                 int depth, int max_depth);
int predict(Node *node, double *features);
void free_tree(Node *node);