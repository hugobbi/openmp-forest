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

/*
    config='p': always parallelize splits
    config='t': parallelize splits only if n >= PARALLEL_SPLIT_THRESHOLD (currently 100, but can be tuned).
                otherwise parallelizes tree node creation
    config='n': paralleize only node creation, never splits (not recommended, just for testing)
    config='P': paralllelize both splits and node creation, no threshold (not recommended, just for testing)
    config='s': sequential, no parallelization (for testing)
*/

#define DEFAULT_NUM_THREADS 1
#define DEFAULT_STRATEGY 'p' // default parallelization strategy: 'p' for splits, 't' for thresholded splits, 
                             // 'n' for node-only, 'P' for always parallel, 's' for sequential
#define TRAIN_DATA  "data/processed/ctype_train.csv"
#define TEST_DATA   "data/processed/ctype_test.csv"

int main(int argc, char *argv[])
{
    // Load data
    printf("Loading training dataset...\n");
    Dataset *ds = csv_read(TRAIN_DATA);
    if (!ds)
    {
        fprintf(stderr, "Failed to load train dataset\n");
        return 1;
    }

    // Setup openmp params
    char config = DEFAULT_STRATEGY; // default parallelization strategy
    if (argc > 1)
    {
        int n_threads = atoi(argv[1]);
        if (n_threads > 0)
        {
            printf("Using %d threads\n", n_threads);
            omp_set_num_threads(n_threads);
        }
        config = (argc > 2) ? argv[2][0] : DEFAULT_STRATEGY;
        printf("Using parallelization strategy: '%c'\n", config);
    }
    else
    {
        printf("Using default %d threads\n", DEFAULT_NUM_THREADS);
        omp_set_num_threads(DEFAULT_NUM_THREADS);
        printf("Using default strategy: '%c'", config);
    }
    if (config != 's' && config != 'n')
    {
        omp_set_max_active_levels(2);
    }
    if (config == 's')
    {
        printf("Running in sequential mode (no parallelization)\n");
        omp_set_num_threads(1);
    }

    // "Train" tree
    printf("Building decision tree...\n");
    int *indices = malloc((size_t)ds->n_samples * sizeof(int));
    if (!indices)
    {
        fprintf(stderr, "Failed to allocate indices for %d samples\n", ds->n_samples);
        dataset_free(ds);
        return 1;
    }
    for (int i = 0; i < ds->n_samples; i++)
        indices[i] = i;
    Node *tree = NULL;
    double build_time = 0.0;
#pragma omp parallel
    {
#pragma omp single
        {
            double t0 = omp_get_wtime();
            tree = build_tree(ds->samples, indices, ds->n_samples, ds->n_features, ds->n_classes, 0, MAX_DEPTH, config);
            build_time = omp_get_wtime() - t0;
        }
    }
    free(indices);

    printf("Tree built in %.4f seconds\n", build_time);
    printf("Tree depth: %d\n", get_tree_depth(tree));

    // Measure accuracy
    printf("Measuring accuracy in test dataset...\n");
    Dataset *test_ds = csv_read(TEST_DATA);
    if (!test_ds)
    {
        fprintf(stderr, "Failed to load test dataset\n");
        free_tree(tree);
        dataset_free(ds);
        return 1;
    }
    double avg_inference_time = 0.0;
    int correct = 0;
    for (int i = 0; i < test_ds->n_samples; i++)
    {
        double t0 = omp_get_wtime();
        int pred = predict(tree, test_ds->samples[i].features);
        double inference_time = omp_get_wtime() - t0;
        avg_inference_time += inference_time;
        if (pred != test_ds->samples[i].label)
        {
            // printf("Sample %d: predicted %d, actual %d\n", i, pred, test_ds->samples[i].label);
        }
        else
        {
            correct++;
        }
    }
    float test_accuracy = (float)correct / test_ds->n_samples;
    avg_inference_time /= test_ds->n_samples;
    printf("Accuracy: %.2f%% (%d/%d correct)\n", test_accuracy * 100, correct, test_ds->n_samples);
    printf("Average inference time: %f seconds\n", avg_inference_time);

    /* Save results to timestamped file */
    if (mkdir("results", 0755) != 0 && errno != EEXIST)
    {
        fprintf(stderr, "Warning: cannot create results directory: %s\n", strerror(errno));
    }
    time_t now = time(NULL);
    struct tm tm = *localtime(&now);
    char results_filename[256];
    strftime(results_filename, sizeof(results_filename), "results/%Y%m%d_%H%M%S_results.txt", &tm);
    FILE *f = fopen(results_filename, "w");
    if (f)
    {
        fprintf(f, "build time=%.6f\n", build_time);
        fprintf(f, "test accuracy=%.6f (%d/%d)\n", test_accuracy, correct, test_ds->n_samples);
        fprintf(f, "avg inference time=%.6f\n", avg_inference_time);
        fprintf(f, "MAX_SAMPLES=%d\nMAX_FEATURES= %d\nMAX_DEPTH= %d\ntree depth=%d\n",
                MAX_SAMPLES, MAX_FEATURES, MAX_DEPTH, get_tree_depth(tree));
        fprintf(f, "threads=%d\nparallelizaiton strategy=%c\n", omp_get_max_threads(), config);
        fclose(f);
        printf("Results saved to %s\n", results_filename);
    }
    else
    {
        fprintf(stderr, "Warning: failed to open results file '%s' for writing\n", results_filename);
    }

    free_tree(tree);
    dataset_free(ds);
    dataset_free(test_ds);

    return 0;
}