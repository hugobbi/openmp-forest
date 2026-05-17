#include "decision_tree.h"
#include "dataset_import.h"
#include <stdio.h>

int main(int argc, char *argv[]) {
    // Example using csv_read:
    //
    //   Dataset *ds = csv_read("data/dataset.csv");
    //   int indices[MAX_SAMPLES];
    //   for (int i = 0; i < ds->n_samples; i++) indices[i] = i;
    //   Node *tree = build_tree(ds->samples, indices, ds->n_samples,
    //                           ds->n_features, ds->n_classes, 0, MAX_DEPTH);
    //   double test_features[] = {5.1, 3.5, 1.4, 0.2};
    //   int pred = predict(tree, test_features);
    //   printf("Predicted class: %d\n", pred);
    //   free_tree(tree);
    //   dataset_free(ds);

    // Hardcoded sample dataset: [income, bought?]
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

    int indices[MAX_SAMPLES];
    for (int i = 0; i < n_samples; i++) indices[i] = i;
    Node *tree = build_tree(data, indices, n_samples, n_features, n_classes, 0, MAX_DEPTH);

    double test_features[] = {55.0};
    int pred = predict(tree, test_features);
    printf("Predicted class for income=55.0: %d\n", pred);

    free_tree(tree);
    return 0;
}