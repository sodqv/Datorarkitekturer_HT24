#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>
#include <pthread.h>
#include <arm_neon.h>

#define NUM_THREADS 4

struct thread_struct {
	float* a; 
	float* b; 
	float* r;
	int num;
};

void* thread_mult_vect(void* arguments);
void* thread_mult_std(void* arg);



void vec_worker_thread(float* a, float* b, float* r, int num)
{
	pthread_t threads[NUM_THREADS];
	int rc;
	
	struct thread_struct arguments[NUM_THREADS]; 

	for (int i = 0; i < NUM_THREADS; i++)
	{

		
		arguments[i].a = &a[(num/NUM_THREADS)*i];
		arguments[i].b = &b[(num/NUM_THREADS)*i];
		arguments[i].r = &r[(num/NUM_THREADS)*i];
		arguments[i].num = num/NUM_THREADS;

//		printf("arguments i: %d:\n a: %p\n b: %p\n r: %p\n", i,arguments[i].a,arguments[i].b,arguments[i].r);

	
		rc = pthread_create(&threads[i], NULL, thread_mult_vect, (void*)&arguments[i]);


		if(rc) {
			printf("Error: Unable to create thread, %d\n", rc);
			exit(-1);
		}
	}

	for (int i = 0; i < NUM_THREADS; i++)
	{
		pthread_join(threads[i], NULL);
	}
}



void std_worker_thread(float* a, float* b, float* r, int num)
{
	pthread_t threads[NUM_THREADS];
	int rc;
	
	struct thread_struct arguments[NUM_THREADS]; 

	for (int i = 0; i < NUM_THREADS; i++)
	{

		
		arguments[i].a = &a[(num/NUM_THREADS)*i];
		arguments[i].b = &b[(num/NUM_THREADS)*i];
		arguments[i].r = &r[(num/NUM_THREADS)*i];
		arguments[i].num = num/NUM_THREADS;

		//printf("arguments i: %d:\n a: %p\n b: %p\n r: %p\n n: %d\n", i,arguments[i].a,arguments[i].b,arguments[i].r, arguments[i].num);

	
		rc = pthread_create(&threads[i], NULL, thread_mult_std, (void*)&arguments[i]);


		if(rc) {
			printf("Error: Unable to create thread, %d\n", rc);
			exit(-1);
		}
	}

	for (int i = 0; i < NUM_THREADS; i++)
	{
		pthread_join(threads[i], NULL);
	}
}





void mult_std(float* a, float* b, float* r, int num)
{
	//printf(" a: %p\n b: %p\n r: %p\n n: %d\n", a,b,r,num);

	for (int i = 0; i < num; i++)
	{
		r[i] = a[i] * b[i];
	}
}


void* thread_mult_std(void* arg)
{
	struct thread_struct arguments = *((struct thread_struct*)arg);
	float* a = arguments.a;
	float* b = arguments.b;
	float* r = arguments.r;
	int num = arguments.num;
#if 1
	for (int i = 0; i < num; i++)
	{
		r[i] = a[i] * b[i];
	}
#else
	mult_std(a, b, r, num);
#endif
	pthread_exit(NULL);
}


void mult_vect(float* a, float* b, float* r, int num)
{
	float32x4_t va, vb, vr;
	for (int i = 0; i < num; i +=4)
	{
		va = vld1q_f32(&a[i]);
		vb = vld1q_f32(&b[i]);
		vr = vmulq_f32(va, vb);
		vst1q_f32(&r[i], vr);
	}
}


void* thread_mult_vect(void* arg)
{
	struct thread_struct arguments = *((struct thread_struct*)arg);
	float* a = arguments.a;
	float* b = arguments.b;
	float* r = arguments.r;
	int num = arguments.num;

	//printf("threads: num: %d:\n a: %p\n b: %p\n r: %p\n", num,a,b,r);

 
	float32x4_t va, vb, vr;
	for (int i = 0; i < num; i +=4)
	{
		va = vld1q_f32(&a[i]);
		vb = vld1q_f32(&b[i]);
		vr = vmulq_f32(va, vb);
		vst1q_f32(&r[i], vr);
	}
	pthread_exit(NULL);
}


int main(int argc, char *argv[]) {
	int num = 100000000;
	float *a = (float*)aligned_alloc(16, num*sizeof(float));
	float *b = (float*)aligned_alloc(16, num*sizeof(float));
	float *r = (float*)aligned_alloc(16, num*sizeof(float));


	struct timespec ts_start;
	struct timespec ts_end_0;
	struct timespec ts_end_1;
	struct timespec ts_end_2;
	struct timespec ts_end_3;
	struct timespec ts_end_4;

	clock_gettime(CLOCK_MONOTONIC, &ts_start);

	for (int i = 0; i < num; i++)
	{
		a[i] = (i % 127)*0.1457f;
		b[i] = (i % 331)*0.1231f;
	}

	mult_std(a, b, r, num);

	
	clock_gettime(CLOCK_MONOTONIC, &ts_end_0);
	mult_std(a, b, r, num);
	clock_gettime(CLOCK_MONOTONIC, &ts_end_1);
	mult_vect(a, b, r, num);
	clock_gettime(CLOCK_MONOTONIC, &ts_end_2);

	vec_worker_thread(a, b, r, num);
	clock_gettime(CLOCK_MONOTONIC, &ts_end_3);

	std_worker_thread(a, b, r, num);
	clock_gettime(CLOCK_MONOTONIC, &ts_end_4);

	double duration_std = (ts_end_1.tv_sec - ts_end_0.tv_sec) + (ts_end_1.tv_nsec - ts_end_0.tv_nsec) * 1e-9;
	double duration_vec = (ts_end_2.tv_sec - ts_end_1.tv_sec) + (ts_end_2.tv_nsec - ts_end_1.tv_nsec) * 1e-9;
	double duration_vec_worker = (ts_end_3.tv_sec - ts_end_2.tv_sec) + (ts_end_3.tv_nsec - ts_end_2.tv_nsec) * 1e-9;
	double duration_std_worker = (ts_end_4.tv_sec - ts_end_3.tv_sec) + (ts_end_4.tv_nsec - ts_end_3.tv_nsec) * 1e-9;
	
	printf("Elapsed time std: %f\n", duration_std);
	printf("Elapsed time vec: %f\n", duration_vec);
	printf("Elapsed time std worker: %f\n", duration_std_worker);
	printf("Elapsed time vec worker: %f\n", duration_vec_worker);
	
	free(a);
	free(b);
	free(r);
	
	return 0;
}

