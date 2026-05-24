#include "decision_tree.h"
#include "dataset_import.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <sys/stat.h>
#include <errno.h>
#include <string.h>
#include <omp.h>

/*
    Label mapping: {'PRAD': 0, 'LUAD': 1, 'BRCA': 2, 'KIRC': 3, 'COAD': 4}
    PRAD 	Prostate adenocarcinoma (prostate)
    LUAD 	Lung adenocarcinoma (lung)
    BRCA 	Breast invasive carcinoma (breast cancer)
    KIRC 	Kidney renal clear cell carcinoma (kidney)
    COAD 	Colon adenocarcinoma (colon)
*/

// k-fold evaluation function 
#define K 5

void k_fold_eval(Dataset *ds, int k) {
    int fold_size = ds->n_samples / k;
    float acc_sum = 0.0;

    for (int fold = 0; fold < k; fold++) {
        int test_start = fold * fold_size;
        int test_end   = test_start + fold_size;

        // Build train indices (everything outside this fold)
        int train_idx[MAX_SAMPLES], n_train = 0;
        for (int i = 0; i < ds->n_samples; i++) {
            if (i < test_start || i >= test_end)
                train_idx[n_train++] = i;
        }

        Node *tree = build_tree(ds->samples, train_idx, n_train,
                                ds->n_features, ds->n_classes, 0, MAX_DEPTH);

        // Evaluate on this fold
        int correct = 0;
        for (int i = test_start; i < test_end; i++) {
            if (predict(tree, ds->samples[i].features) == ds->samples[i].label)
                correct++;
        }

        float acc = (float)correct / fold_size;
        printf("Fold %d: %.4f (%d/%d)\n", fold + 1, acc, correct, fold_size);
        acc_sum += acc;
        free_tree(tree);
    }

    printf("Mean accuracy: %.4f\n", acc_sum / k);
}

#define DEFAULT_NUM_THREADS 1

int main(int argc, char *argv[]) 
{    
    // Load data
    printf("Loading training dataset...\n");
    Dataset *ds = csv_read("data/processed/train.csv");
    if (!ds) {
        fprintf(stderr, "Failed to load train dataset\n");
        return 1;
    }

    // Setup openmp params
    if (argc > 1) {
        int n_threads = atoi(argv[1]);
        if (n_threads > 0) {
            printf("Using %d threads\n", n_threads);
            omp_set_num_threads(n_threads);
        }
    } else {
        printf("Using default %d threads\n", DEFAULT_NUM_THREADS);
        omp_set_num_threads(DEFAULT_NUM_THREADS);
    }

    // "Train" tree
    printf("Building decision tree...\n");
    int indices[MAX_SAMPLES];
    for (int i = 0; i < ds->n_samples; i++) indices[i] = i;
    double t0 = omp_get_wtime();
    Node *tree = build_tree(ds->samples, indices, ds->n_samples,
                            ds->n_features, ds->n_classes, 0, MAX_DEPTH);
    double build_time = omp_get_wtime() - t0;
    printf("Tree built in %.4f seconds\n", build_time);
    
    // Measure accuracy
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
        t0 = omp_get_wtime();
        int pred = predict(tree, test_ds->samples[i].features);
        double inference_time = omp_get_wtime() - t0;
        avg_inference_time += inference_time;
        if (pred != test_ds->samples[i].label) {
            printf("Sample %d: predicted %d, actual %d\n", i, pred, test_ds->samples[i].label);
        } else {
            correct++;
        }
    }
    float test_accuracy = (float)correct / test_ds->n_samples;
    avg_inference_time /= test_ds->n_samples;
    printf("Accuracy: %.2f%% (%d/%d correct)\n", test_accuracy * 100, correct, test_ds->n_samples);
    printf("Average inference time: %f seconds\n", avg_inference_time);

    /* Save results to timestamped file */
    if (mkdir("results", 0755) != 0 && errno != EEXIST) {
        fprintf(stderr, "Warning: cannot create results directory: %s\n", strerror(errno));
    }
    time_t now = time(NULL);
    struct tm tm = *localtime(&now);
    char results_filename[256];
    strftime(results_filename, sizeof(results_filename), "results/%Y%m%d_%H%M%S_results.txt", &tm);
    FILE *f = fopen(results_filename, "w");
    if (f) {
        fprintf(f, "build time=%.6f\n", build_time);
        fprintf(f, "test accuracy=%.6f (%d/%d)\n", test_accuracy, correct, test_ds->n_samples);
        fprintf(f, "avg inference time=%.6f\n", avg_inference_time);
        fprintf(f, "MAX_SAMPLES=%d\nMAX_FEATURES= %d\nMAX_DEPTH= %d\n",
                MAX_SAMPLES, MAX_FEATURES, MAX_DEPTH);
        fprintf(f, "threads=%d\n", omp_get_max_threads());
        fclose(f);
        printf("Results saved to %s\n", results_filename);
    } else {
        fprintf(stderr, "Warning: failed to open results file '%s' for writing\n", results_filename);
    }

    free_tree(tree);
    dataset_free(ds);
    dataset_free(test_ds);
    
    return 0;
}