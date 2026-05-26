#include "dataset_import.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>

#define CSV_LINE_MAX 4096

// Returns the number of delimiter occurrences in s (used to count columns).
static int count_delim(const char *s, char delim) {
    int count = 0;
    while (*s) {
        if (*s == delim) count++;
        s++;
    }
    return count;
}

// Strips trailing newline / carriage-return characters in place.
static void strip_newline(char *s) {
    size_t len = strlen(s);
    while (len > 0 && (s[len - 1] == '\n' || s[len - 1] == '\r')) {
        s[--len] = '\0';
    }
}

Dataset *csv_read(const char *path) {
    return csv_read_delim(path, ',');
}

Dataset *csv_read_delim(const char *path, char delim) {
    FILE *fp = fopen(path, "r");
    if (!fp) {
        fprintf(stderr, "csv_read: cannot open '%s': %s\n", path, strerror(errno));
        return NULL;
    }

    char *line = NULL;
    size_t linecap = 0;
    ssize_t linelen;

    // Read header row 
    linelen = getline(&line, &linecap, fp);
    if (linelen == -1) {
        fprintf(stderr, "csv_read: '%s' is empty or cannot read header\n", path);
        fclose(fp);
        free(line);
        return NULL;
    }
    strip_newline(line);
    int n_cols = count_delim(line, delim) + 1;
    int n_features = n_cols - 1; // last column is the label

    if (n_features <= 0) {
        fprintf(stderr, "csv_read: need at least 2 columns (1 feature + 1 label)\n");
        fclose(fp);
        return NULL;
    }
    if (n_features > MAX_FEATURES) {
        fprintf(stderr, "csv_read: %d features exceeds MAX_FEATURES (%d)\n",
                n_features, MAX_FEATURES);
        fclose(fp);
        return NULL;
    }

    // Allocate samples up to MAX_SAMPLES.
    Sample *samples = malloc(MAX_SAMPLES * sizeof(Sample));
    if (!samples) {
        fprintf(stderr, "csv_read: out of memory\n");
        fclose(fp);
        return NULL;
    }

    int n_samples = 0;
    int n_classes = 0; // track max label seen + 1
    int row = 1;       // row counter for error messages (1-indexed, after header)

    while ((linelen = getline(&line, &linecap, fp)) != -1) {
        row++;
        strip_newline(line);
        if (line[0] == '\0') continue; /* skip blank lines */

        if (n_samples >= MAX_SAMPLES) {
            fprintf(stderr, "csv_read: warning: row %d exceeds MAX_SAMPLES (%d), truncating\n",
                    row, MAX_SAMPLES);
            break;
        }

        Sample *s = &samples[n_samples];
        char *cursor = line;
        char *end;

        // Parse feature columns.
        for (int f = 0; f < n_features; f++) {
            errno = 0;
            s->features[f] = strtod(cursor, &end);
            if (end == cursor || errno != 0) {
                fprintf(stderr, "csv_read: row %d col %d: invalid number\n", row, f + 1);
                free(samples);
                fclose(fp);
                free(line);
                return NULL;
            }
            if (*end != delim && f < n_features - 1) {
                fprintf(stderr, "csv_read: row %d: expected %d columns, got fewer\n",
                        row, n_cols);
                free(samples);
                fclose(fp);
                free(line);
                return NULL;
            }
            cursor = end + 1; // step past delimiter
        }

        // Parse label column (last column, must be a non-negative integer).
        errno = 0;
        long label = strtol(cursor, &end, 10);
        if (end == cursor || errno != 0 || label < 0) {
            fprintf(stderr, "csv_read: row %d: invalid label '%s'\n", row, cursor);
            free(samples);
            fclose(fp);
            free(line);
            return NULL;
        }
        s->label = (int)label;
        if (s->label >= MAX_CLASSES) {
            fprintf(stderr, "csv_read: row %d: label %d exceeds MAX_CLASSES (%d)\n",
                row, s->label, MAX_CLASSES);
            free(samples);
            fclose(fp);
            free(line);
            return NULL;
        }
        if (s->label + 1 > n_classes) n_classes = s->label + 1;

        n_samples++;
    }

    fclose(fp);
    free(line);
    if (n_samples == 0) {
        fprintf(stderr, "csv_read: no data rows found in '%s'\n", path);
        free(samples);
        return NULL;
    }

    Dataset *ds = malloc(sizeof(Dataset));
    if (!ds) {
        fprintf(stderr, "csv_read: out of memory\n");
        free(samples);
        return NULL;
    }
    ds->samples   = samples;
    ds->n_samples = n_samples;
    ds->n_features = n_features;
    ds->n_classes  = n_classes;
    return ds;
}

void dataset_free(Dataset *ds) {
    if (!ds) return;
    free(ds->samples);
    free(ds);
}
