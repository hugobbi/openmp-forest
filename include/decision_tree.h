#ifndef DECISION_TREE_H
#define DECISION_TREE_H

#define MAX_SAMPLES 600000 // 1000
#define MAX_FEATURES 60 // 25000
#define MAX_CLASSES 100
#define MAX_DEPTH 100

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
BestSplit find_best_split_parallel(Sample *data, int *indices, int n,
                          int n_features, int n_classes);
BestSplit find_best_split_sequential(Sample *data, int *indices, int n,
                          int n_features, int n_classes);
Node *build_tree(Sample *data, int *indices, int n,
                 int n_features, int n_classes,
                 int depth, int max_depth, char config);
int predict(Node *node, double *features);
int get_tree_depth(const Node *node);
void free_tree(Node *node);

#endif // DECISION_TREE_H