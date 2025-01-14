#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <windows.h>
#include <share.h>

#define THREAD_COUNT 12
#define EPSILON 0.0000001f

typedef struct {
    int index;
    float distance;
} IndexDistance;

typedef struct {
    int k;
    float distanceThreshold;
    float distanceExponent;
} KnnParameters;

typedef struct {
    FILE* resultsFile;
    KnnParameters* knnParameters;
    int knnParametersIndex;
    int knnParametersCount;
    HANDLE parametersLock;
    HANDLE resultsLock;
    int trainCount;
    int testCount;
    int inputSize;
    int outputSize;
    float* trainInputs;
    float* trainOutputs;
    int* trainArgmax;
    float* testInputs;
    float* testOutputs;
    int* testArgmax;
} ThreadArgs;

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
        float weight = 1.0f / (distance + EPSILON);
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

FILE* createResultsFile(char* filename)
{
    FILE* file = _fsopen(filename, "w", _SH_DENYNO);
    if (file == NULL)
    {
        printf("Could not create file %s\n", filename);
        exit(1);
    }
    fprintf(file, "K,DistanceThreshold,DistanceExponent,CorrectCount\n");
    return file;
}

DWORD WINAPI threadEntry(LPVOID arg) 
{
    ThreadArgs* threadArgs = (ThreadArgs*)arg;

    IndexDistance* indexDistances = (IndexDistance*)calloc(threadArgs->trainCount, sizeof(IndexDistance));
    if (indexDistances == NULL) 
    {
        printf("Failed to allocate memory for index distances.\n");
        exit(1);
    }

    float* predictionOutput = (float*)calloc(threadArgs->outputSize, sizeof(float));
    if (predictionOutput == NULL) 
    {
        printf("Failed to allocate memory for prediction output.\n");
        exit(1);
    }

    // loop till complete
    for (;;)
    {
        // lock parameters
        WaitForSingleObject(threadArgs->parametersLock, INFINITE);

        // get index
        int knnParametersIndex = threadArgs->knnParametersIndex;

        // if we are done break
        if (knnParametersIndex >= threadArgs->knnParametersCount)
        {
            ReleaseMutex(threadArgs->parametersLock);
            break;
        }

        // get parameters
        KnnParameters knnParameters = threadArgs->knnParameters[knnParametersIndex];

        // increment index
        threadArgs->knnParametersIndex++;

        // release parameters
        ReleaseMutex(threadArgs->parametersLock);

        // test knn
        int correctCount = knnTest(threadArgs->inputSize, threadArgs->outputSize, threadArgs->trainCount, threadArgs->trainInputs, threadArgs->trainOutputs, threadArgs->testCount, threadArgs->testInputs, threadArgs->testArgmax, predictionOutput, indexDistances, knnParameters.k, knnParameters.distanceThreshold, knnParameters.distanceExponent);

        // lock results
        WaitForSingleObject(threadArgs->resultsLock, INFINITE);

        // write results
        fprintf(threadArgs->resultsFile, "%d,%f,%f,%d\n", knnParameters.k, knnParameters.distanceThreshold, knnParameters.distanceExponent, correctCount);

        // flush
        fflush(threadArgs->resultsFile);

        // console log results
        printf("K: %d, DistanceThreshold: %f, DistanceExponent: %f, CorrectCount: %d\n", knnParameters.k, knnParameters.distanceThreshold, knnParameters.distanceExponent, correctCount);

        // release results
        ReleaseMutex(threadArgs->resultsLock);
    }

    return 0;
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

    int kMin = 1;
    int kMax = 20;
    int kStep = 1;
    float distanceThresholdMin = 0.000f;
    float distanceThresholdMax = 1.000f;
    float distanceThresholdStep = 0.005f;
    float distanceExponentMin = 0.01f;
    float distanceExponentMax = 20.0f;
    float distanceExponentStep = 0.01f;
    int knnParametersCount = 0;
    for (int k = kMin; k <= kMax; k += kStep) 
    {
        for (float distanceThreshold = distanceThresholdMin; distanceThreshold <= distanceThresholdMax; distanceThreshold += distanceThresholdStep) 
        {
            for (float distanceExponent = distanceExponentMin; distanceExponent <= distanceExponentMax; distanceExponent += distanceExponentStep) 
            {
                knnParametersCount++;
            }
        }
    }
    printf("KNN Parameters Count: %d\n", knnParametersCount);

    KnnParameters* knnParameters = (KnnParameters*)calloc(knnParametersCount, sizeof(KnnParameters));
    if (knnParameters == NULL) 
    {
        printf("Failed to allocate memory for knn parameters.\n");
        exit(1);
    }

    int combinationIndex = 0;
    for (int k = kMin; k <= kMax; k += kStep) 
    {
        for (float distanceThreshold = distanceThresholdMin; distanceThreshold <= distanceThresholdMax; distanceThreshold += distanceThresholdStep) 
        {
            for (float distanceExponent = distanceExponentMin; distanceExponent <= distanceExponentMax; distanceExponent += distanceExponentStep) 
            {
                knnParameters[combinationIndex].k = k;
                knnParameters[combinationIndex].distanceThreshold = distanceThreshold;
                knnParameters[combinationIndex].distanceExponent = distanceExponent;
                combinationIndex++;
            }
        }
    }

    FILE* resultsFile = createResultsFile("./results.csv");
    
    HANDLE parametersLock = CreateMutex(NULL, FALSE, NULL);
    HANDLE resultsLock = CreateMutex(NULL, FALSE, NULL);
    if (parametersLock == NULL || resultsLock == NULL) 
    {
        printf("Failed to create mutexes.\n");
        exit(1);
    }

    ThreadArgs* threadArgs = (ThreadArgs*)calloc(1, sizeof(ThreadArgs));
    if (threadArgs == NULL) 
    {
        printf("Failed to allocate memory for thread args.\n");
        exit(1);
    }
    threadArgs->resultsFile = resultsFile;
    threadArgs->knnParameters = knnParameters;
    threadArgs->knnParametersIndex = 0;
    threadArgs->knnParametersCount = knnParametersCount;
    threadArgs->parametersLock = parametersLock;
    threadArgs->resultsLock = resultsLock;
    threadArgs->trainCount = trainCount;
    threadArgs->testCount = testCount;
    threadArgs->inputSize = inputSize;
    threadArgs->outputSize = outputSize;
    threadArgs->trainInputs = trainInputs;
    threadArgs->trainOutputs = trainOutputs;
    threadArgs->trainArgmax = trainArgmax;
    threadArgs->testInputs = testInputs;
    threadArgs->testOutputs = testOutputs;
    threadArgs->testArgmax = testArgmax;

    HANDLE threads[THREAD_COUNT];
    for (int threadIndex = 0; threadIndex < THREAD_COUNT; threadIndex++) {
        threads[threadIndex] = CreateThread(NULL, 0, threadEntry, threadArgs, 0, NULL);
        if (threads[threadIndex] == NULL) {
            perror("Failed to create thread");
            exit(1);
        }
    }
    WaitForMultipleObjects(THREAD_COUNT, threads, TRUE, INFINITE);
    fclose(resultsFile);
    return 0;
}