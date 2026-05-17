#ifndef DATASET_IMPORT_H
#define DATASET_IMPORT_H

#include "decision_tree.h"

typedef struct {
    Sample *samples;
    int     n_samples;
    int     n_features;
    int     n_classes;
} Dataset;

// Reads a CSV file where:
//   - first row is a header (skipped)
//   - each subsequent row has n_features numeric columns followed by an integer label column
//   - delimiter is ',' by default; use csv_read_delim for other delimiters
//
// Returns a heap-allocated Dataset on success (caller must call dataset_free).
// Returns NULL on error (file not found, malformed row, exceeds MAX_SAMPLES/MAX_FEATURES).
Dataset *csv_read(const char *path);
Dataset *csv_read_delim(const char *path, char delim);

void dataset_free(Dataset *ds);

#endif // DATASET_IMPORT_H
