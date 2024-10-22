#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <arm_neon.h>
#include <pthread.h>
#include <unistd.h>


typedef struct {
    float* a;
    float* b;
    float* r;
    int start_index;
    int end_index;
} ThreadData;



//standard code
void* worker_mult_std(void* arg)
{
    ThreadData* data = (ThreadData*)arg;

    float* a = data -> a;               //get pointer to the a-array from the ThreadData struct
    float* b = data -> b;
    float* r = data -> r;

    int start = data -> start_index;    //get pointers to start and end indices
    int end = data -> end_index;


    //multiply a * b
    for (int i = start; i < end; i++)
    {
        r[i] = a[i] * b[i];
    }


    pthread_exit(NULL);
}



//vectorized code
void* worker_mult_vec(void* arg)
{
    ThreadData* data = (ThreadData*)arg;

    float* a = data -> a;
    float* b = data -> b;
    float* r = data -> r;

    int start = data -> start_index;
    int end = data -> end_index;

    float32x4_t va, vb, vr;     //declare the vector registers for SIMD processing

    //4 floats are processed at a time by using SIMD
    //loop through the array in steps of 4 elements
    for (int i = start; i < end; i += 4)
    {
        va = vld1q_f32( &a[i] );    //loads 4 floats from the a-array, starting at index i
        vb = vld1q_f32( &b[i] );

        vr = vmulq_f32( &r[i] );    //performs vector multiplication on the loaded values

        vst1q_f32( &r[i], vr );     //stores the result in the r-array
    }


    pthread_exit(NULL);
}







int main(int argc, char *argv[])
{
    int num = 100000000;

    int NUM_THREADS = sysconf(_SC_NPROCESSORS_ONLN);    //find the number of available CPU cores

    //the optimization is the second argument from the command line
    const char* optimization_level = argv[1];




    //allocates aligned memory for the arrays
    float *a = (float*)aligned_alloc(16, num*sizeof(float));
    float *b = (float*)aligned_alloc(16, num*sizeof(float));

    float *r = (float*)aligned_alloc(16, num*sizeof(float));



    //array initialization
    for (int i = 0; i < num; i++)
    {
        a[i] = (i % 127)*1.1457f;
        b[i] = (i % 331)*0.1231f;
    }



    pthread_t threads[NUM_THREADS]; //create the threads

    ThreadData thread_data[NUM_THREADS];


    struct timespec ts_start;
    struct timespec ts_end_1;
    struct timespec ts_end_2;




    // standard multiplication
    clock_gettime(CLOCK_MONOTONIC, &ts_start);

    for( int t = 0; t < NUM_THREADS; t++)
    {
        thread_data[t].a = a;
        thread_data[t].b = b;
        thread_data[t].r = r;

        thread_data[t].start_index = t * (num / NUM_THREADS);
        thread_data[t].end_index = (t + 1) * (num / NUM_THREADS);

        pthread_create(&threads[t], NULL, worker_mult_std, (void*)&thread_data[t]);
    }


    for (int t = 0; t < NUM_THREADS; t++)
    {
        pthread_join(threads[t], NULL);         //join the threads
    }

    clock_gettime(CLOCK_MONOTONIC, &ts_end_1);
    double duration_std = (ts_end_1.tv_sec - ts_start.tv_sec) + (ts_end_1.tv_nsec - ts_start.tv_nsec) * 1e-9;





    //vector multiplication
    clock_gettime(CLOCK_MONOTONIC, &ts_start);

    for( int t = 0; t < NUM_THREADS; t++)
    {
        thread_data[t].a = a;
        thread_data[t].b = b;
        thread_data[t].r = r;

        thread_data[t].start_index = t * (num / NUM_THREADS);
        thread_data[t].end_index = (t + 1) * (num / NUM_THREADS);

        pthread_create(&threads[t], NULL, worker_mult_vec, (void*)&thread_data[t]);
    }

    
    for (int t = 0; t < NUM_THREADS; t++)
    {
        pthread_join(threads[t], NULL);         //join the threads
    }

    clock_gettime(CLOCK_MONOTONIC, &ts_end_2);
    double duration_vec = (ts_end_2.tv_sec - ts_end_1.tv_sec) + (ts_end_2.tv_nsec - ts_end_1.tv_nsec) * 1e-9;






    //prints the table header only when the optimization is O0 (on the first run)
    if (strcmp(optimization_level, "O0") == 0)
    {
        printf("| Optimization Level | Elapsed Time (std) | Elapsed Time (vec) |\n");
        printf("| ------------------ | ------------------ | ------------------ |\n");

    }

    //prints the results in the table
    printf("| %s                 | %f                 | %f                 |\n", optimization_level, duration_std, duration_vec);




    free(a);
    free(b);
    free(r);

    return 0;
}

