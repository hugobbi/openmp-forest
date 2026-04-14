#include "decision_tree.h"
#include <stdio.h>

int main(int argc, char *argv[]) {
    // Sample dataset: [income, bought?]
    Sample data[] = {
        {{30.0}, 0},
        {{40.0}, 0},
        {{50.0}, 1},
        {{60.0}, 1},
        {{70.0}, 1}
    };
    int n_samples = sizeof(data) / sizeof(Sample);
    int n_features = 1;
    int n_classes = 2;

    // Build tree
    int indices[MAX_SAMPLES];
    for (int i = 0; i < n_samples; i++) indices[i] = i;
    Node *tree = build_tree(data, indices, n_samples, n_features, n_classes, 0, MAX_DEPTH);

    // Predict
    double test_features[] = {55.0};
    int pred = predict(tree, test_features);
    printf("Predicted class for income=55.0: %d\n", pred);

    // Free memory
    free_tree(tree);
    return 0;
}