#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

typedef struct {
    int index;
    float distance;
} IndexDistance;

char* strsep(char** stringp, const char* delim) 
{
    if (*stringp == NULL)
    {
        return NULL;
    }

    char* start = *stringp;
    char* end = strpbrk(start, delim);

    if (end) 
    {
        *end = '\0';
        *stringp = end + 1;
    } 
    else 
    {
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
    if (fopen_s(&file, filename, "r") != 0) 
    {
        printf("Could not open file %s\n", filename);
        exit(1);
    }

    // allocate memory for inputs
    *inputs = (float*)calloc(count * inputSize, sizeof(float));
    if (*inputs == NULL)
    {
        printf("Could not allocate memory for inputs\n");
        fclose(file);
        exit(1);
    }

    // allocate memory for outputs
    *outputs = (float*)calloc(count * outputSize, sizeof(float));
    if (*outputs == NULL)
    {
        printf("Could not allocate memory for outputs\n");
        fclose(file);
        exit(1);
    }

    // skip the header row
    if (fgets(line, sizeof(line), file) == NULL)
    {
        printf("File is empty or not formatted correctly\n");
        fclose(file);
        exit(1);
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
                    fclose(file);
                    exit(1);
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
            fclose(file);
            exit(1);
        }

        row++;
    }

    fclose(file);
    return 0;
}

int argmax(int size, float* values)
{
    int maxIndex = 0;
    float maxValue = values[0];
    for (int i = 1; i < size; i++)
    {
        if (values[i] > maxValue)
        {
            maxIndex = i;
            maxValue = values[i];
        }
    }
    return maxIndex;
}

int compareIndexDistance(const void* a, const void* b)
{
    IndexDistance* id1 = (IndexDistance*)a;
    IndexDistance* id2 = (IndexDistance*)b;
    if (id1->distance < id2->distance) 
    {
        return -1;
    }
    if (id1->distance > id2->distance)
    {
        return 1;
    }
    return 0;
}

int knn(int inputSize, int outputSize, int trainCount, float* trainInputs, float* trainOutputs, float* testInput, float* predictionOutput, IndexDistance* indexDistances, int k, float distanceThreshold, float distanceExponent)
{
    // calculate distances between test input and train inputs
    for (int trainIndex = 0; trainIndex < trainCount; trainIndex++)
    {
        float distance = 0.0f;
        for (int inputIndex = 0; inputIndex < inputSize; inputIndex++)
        {
            float difference = fabs(testInput[inputIndex] - trainInputs[trainIndex * inputSize + inputIndex]);
            if (difference <= distanceThreshold)
            {
                continue;
            }
            distance += pow(difference, distanceExponent);
        }
        indexDistances[trainIndex].index = trainIndex;
        indexDistances[trainIndex].distance = distance;
    }

    // sort low to high distance
    qsort(indexDistances, trainCount, sizeof(IndexDistance), compareIndexDistance);

    // zero prediction output
    memset(predictionOutput, 0, outputSize * sizeof(float));

    // accumulate
    float weightSum = 0.0f;
    for (int neighbourIndex = 0; neighbourIndex < k && neighbourIndex < trainCount; neighbourIndex++)
    {
        int trainIndex = indexDistances[neighbourIndex].index;
        float distance = indexDistances[neighbourIndex].distance;
        float weight = 1.0f / (distance + 0.0000001f);
        weightSum += weight;
        for (int outputIndex = 0; outputIndex < outputSize; outputIndex++)
        {
            float outputValue = trainOutputs[trainIndex * outputSize + outputIndex];
            predictionOutput[outputIndex] += outputValue * weight;
        }
    }

    // normalize
    for (int outputIndex = 0; outputIndex < outputSize; outputIndex++)
    {
        predictionOutput[outputIndex] /= weightSum;
    }

    // done
    return 0;
}

int knnTest(int inputSize, int outputSize, int trainCount, float* trainInputs, float* trainOutputs, int testCount, float* testInputs, int* testArgmax, float* predictionOutput, IndexDistance* indexDistances, int k, float distanceThreshold, float distanceExponent)
{
    int correctCount = 0;
    for (int testIndex = 0; testIndex < testCount; testIndex++)
    {
        knn(inputSize, outputSize, trainCount, trainInputs, trainOutputs, &testInputs[testIndex * inputSize], predictionOutput, indexDistances, k, distanceThreshold, distanceExponent);
        int predictionArgmax = argmax(outputSize, predictionOutput);
        int testArgmaxValue = testArgmax[testIndex];
        if (predictionArgmax == testArgmaxValue)
        {
            correctCount++;
        }
    }
    return correctCount;
}

int main() 
{
    int result = 0;
    int trainCount = 1000;
    int testCount = 1000;
    int inputSize = 784;
    int outputSize = 10;

    float* trainInputs = NULL;
    float* trainOutputs = NULL;
    int* trainArgmax = NULL;
    float* testInputs = NULL;
    float* testOutputs = NULL;
    int* testArgmax = NULL;

    IndexDistance* indexDistances = NULL;
    float* predictionOutput = NULL;

    result = loadMNIST("d:/data/mnist_train.csv", trainCount, inputSize, outputSize, &trainInputs, &trainOutputs);
    if (result != 0) 
    {
        printf("Failed to load training data.\n");
        exit(1);
    }

    trainArgmax = (int*)calloc(trainCount, sizeof(int));
    if (trainArgmax == NULL) 
    {
        printf("Failed to allocate memory for training argmax.\n");
        exit(1);
    }

    for (int trainIndex = 0; trainIndex < trainCount; trainIndex++) 
    {
        trainArgmax[trainIndex] = argmax(outputSize, &trainOutputs[trainIndex * outputSize]);
    }

    result = loadMNIST("d:/data/mnist_test.csv", testCount, inputSize, outputSize, &testInputs, &testOutputs);
    if (result != 0) 
    {
        printf("Failed to load test data.\n");
        exit(1);
    }

    testArgmax = (int*)calloc(testCount, sizeof(int));
    if (testArgmax == NULL) 
    {
        printf("Failed to allocate memory for test argmax.\n");
        exit(1);
    }

    for (int testIndex = 0; testIndex < testCount; testIndex++) {
        testArgmax[testIndex] = argmax(outputSize, &testOutputs[testIndex * outputSize]);
    }

    indexDistances = (IndexDistance*)calloc(trainCount, sizeof(IndexDistance));
    if (indexDistances == NULL) 
    {
        printf("Failed to allocate memory for index distances.\n");
        exit(1);
    }

    predictionOutput = (float*)calloc(outputSize, sizeof(float));
    if (predictionOutput == NULL) 
    {
        printf("Failed to allocate memory for prediction output.\n");
        exit(1);
    }

    int k = 3;
    float distanceThreshold = 0.1f;
    float distanceExponent = 5.0f;

    int correctCount = knnTest(inputSize, outputSize, trainCount, trainInputs, trainOutputs, testCount, testInputs, testArgmax, predictionOutput, indexDistances, k, distanceThreshold, distanceExponent);
    printf("Correct count: %d of: %d\n", correctCount, testCount);

    return 0;
}