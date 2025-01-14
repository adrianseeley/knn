#include <stdio.h>
#include <stdlib.h>
#include <string.h>

char* strsep(char** stringp, const char* delim) {
    if (*stringp == NULL) {
        return NULL;
    }

    char* start = *stringp;
    char* end = strpbrk(start, delim);

    if (end) {
        *end = '\0';
        *stringp = end + 1;
    } else {
        *stringp = NULL;
    }

    return start;
}

int loadMNIST(const char* filename, int count, int inputSize, int outputSize, float** inputs, float** outputs)
{
    char line[20000];
    int row = 0;

    // open the file
    FILE* file;
    if (fopen_s(&file, filename, "r") != 0) {
        printf("Could not open file %s\n", filename);
        return 1;
    }

    // allocate memory for inputs
    *inputs = (float*)calloc(count * inputSize, sizeof(float));
    if (*inputs == NULL)
    {
        printf("Could not allocate memory for inputs\n");
        fclose(file);
        return 1;
    }

    // allocate memory for outputs
    *outputs = (float*)calloc(count * outputSize, sizeof(float));
    if (*outputs == NULL)
    {
        printf("Could not allocate memory for outputs\n");
        free(*inputs);
        fclose(file);
        return 1;
    }

    // skip the header row
    if (fgets(line, sizeof(line), file) == NULL)
    {
        printf("File is empty or not formatted correctly\n");
        free(*inputs);
        free(*outputs);
        fclose(file);
        return 1;
    }

    // read rows
    while (fgets(line, sizeof(line), file) && row < count)
    {
        char* tmp = line;
        char* field = strsep(&tmp, ",");
        int col = 0;

        while (field)
        {
            // first column is the label
            if (col == 0) 
            {
                int label = atoi(field);
                if (label < 0 || label >= outputSize)
                {
                    printf("Invalid label value: %d\n", label);
                    free(*inputs);
                    free(*outputs);
                    fclose(file);
                    return 1;
                }
                (*outputs)[row * outputSize + label] = 1.0f;
            }
            // remaining columns are input values
            else if (col - 1 < inputSize) 
            {
                (*inputs)[row * inputSize + (col - 1)] = atof(field) / 255.0f;
            }
            field = strsep(&tmp, ",");
            col++;
        }

        // validate input column count
        if (col - 1 != inputSize) 
        {
            printf("Invalid number of input columns at row %d\n", row + 1);
            printf("Expected: %d, Actual: %d\n", inputSize, col - 1);
            free(*inputs);
            free(*outputs);
            fclose(file);
            return 1;
        }

        row++;
    }

    fclose(file);
    return 0;
}

int main() {
    int result = 0;
    int trainCount = 1000;
    int testCount = 1000;
    int inputSize = 784;
    int outputSize = 10;

    float* trainInputs = NULL;
    float* trainOutputs = NULL;
    float* testInputs = NULL;
    float* testOutputs = NULL;

    result = loadMNIST("d:/data/mnist_train.csv", trainCount, inputSize, outputSize, &trainInputs, &trainOutputs);
    if (result != 0) {
        printf("Failed to load training data.\n");
        return result;
    }

    result = loadMNIST("d:/data/mnist_test.csv", testCount, inputSize, outputSize, &testInputs, &testOutputs);
    if (result != 0) {
        printf("Failed to load test data.\n");
        free(trainInputs);
        free(trainOutputs);
        return result;
    }

    // show the first 10 training and test labels
    for (int i = 0; i < 10; i++) {
        printf("Train Label %d: ", i);
        for (int j = 0; j < outputSize; j++) {
            printf("%d", (int)trainOutputs[i * outputSize + j]);
        }
        printf("\n");

        printf("Test Label %d: ", i);
        for (int j = 0; j < outputSize; j++) {
            printf("%d", (int)testOutputs[i * outputSize + j]);
        }
        printf("\n");
    }

    // show the first 10 training and test inputs
    for (int i = 0; i < 10; i++) {
        printf("Train Input %d: ", i);
        for (int j = 0; j < inputSize; j++) {
            printf("%f ", trainInputs[i * inputSize + j]);
        }
        printf("\n");

        printf("Test Input %d: ", i);
        for (int j = 0; j < inputSize; j++) {
            printf("%f ", testInputs[i * inputSize + j]);
        }
        printf("\n");
    }

    return result;
}