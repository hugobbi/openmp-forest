#include "decision_tree.h"
#include "dataset_import.h"
#include <stdio.h>
#include <time.h>

/*
    Label mapping: {'PRAD': 0, 'LUAD': 1, 'BRCA': 2, 'KIRC': 3, 'COAD': 4}
    PRAD 	Prostate adenocarcinoma (prostate)
    LUAD 	Lung adenocarcinoma (lung)
    BRCA 	Breast invasive carcinoma (breast cancer)
    KIRC 	Kidney renal clear cell carcinoma (kidney)
    COAD 	Colon adenocarcinoma (colon)
*/

int main() 
{    
    printf("Loading training dataset...\n");
    Dataset *ds = csv_read("data/processed/train.csv");
    if (!ds) {
        fprintf(stderr, "Failed to load train dataset\n");
        return 1;
    }

    printf("Building decision tree...\n");
    int indices[MAX_SAMPLES];
    for (int i = 0; i < ds->n_samples; i++) indices[i] = i;
    clock_t start = clock();
    Node *tree = build_tree(ds->samples, indices, ds->n_samples,
                            ds->n_features, ds->n_classes, 0, MAX_DEPTH);
    clock_t end = clock();
    double time_spent = (double)(end - start) / CLOCKS_PER_SEC;
    printf("Tree built in %.4f seconds\n", time_spent);

    printf("Measuring accuracy in test dataset...\n");
    Dataset *test_ds = csv_read("data/processed/test.csv");
    if (!test_ds) {
        fprintf(stderr, "Failed to load test dataset\n");
        free_tree(tree);
        dataset_free(ds);
        return 1;
    }
    double avg_inference_time = 0.0;
    int correct = 0;
    for (int i = 0; i < test_ds->n_samples; i++) {
        start = clock();
        int pred = predict(tree, test_ds->samples[i].features);
        end = clock();
        time_spent = (double)(end - start) / CLOCKS_PER_SEC;
        avg_inference_time += time_spent;
        if (pred != test_ds->samples[i].label) {
            printf("Sample %d: predicted %d, actual %d\n", i, pred, test_ds->samples[i].label);
        } else {
            correct++;
        }
    }
    float accuracy = (float)correct / test_ds->n_samples;
    avg_inference_time /= test_ds->n_samples;
    printf("Accuracy: %.2f%% (%d/%d correct)\n", accuracy * 100, correct, test_ds->n_samples);
    printf("Average inference time: %f seconds\n", avg_inference_time);

    free_tree(tree);
    dataset_free(ds);
    dataset_free(test_ds);
    
    return 0;
}