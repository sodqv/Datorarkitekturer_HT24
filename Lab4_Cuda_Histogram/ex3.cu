#include <jetson-utils/videoSource.h>
#include <jetson-utils/videoOutput.h>
#include <iostream>
#include <math.h>

__device__ unsigned char grayScale(unsigned char r,unsigned char g,unsigned char b);


__global__ void calcHistogramKernel(uchar4* image, int* hist_t, int width, int height)
{
	__shared__ int histo_local[256];

	

	histo_local[threadIdx.x] = 0;		

	__syncthreads();


	int index = blockIdx.x * blockDim.x + threadIdx.x;
	int stride = blockDim.x * gridDim.x;

	for (int i = index; i < (width*height); i+= stride) {
		
//		unsigned char grey = grayScale(image[i].x, image[i].y, image[i].z);
		unsigned char grey = image[i].x;
		
		atomicAdd(&(histo_local[grey]), 1); 
	}
	
	__syncthreads();


	atomicAdd(&(hist_t[threadIdx.x]), histo_local[threadIdx.x]);

	__syncthreads();
}






__global__ void plotHistogramKernel(uchar4* image, int* histogram, int width, int height, int max_freq)
{
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	uchar4 white_pixel = make_uchar4(255, 255, 255, 255);
	uchar4 black_pixel = make_uchar4(0, 0, 0, 255);

	if (index < 256)
	{
		int freq = histogram[index] * 256 / max_freq;
		for ( int i = 0; i < 256; i++ )
		{
			int row = height - i - 1;
			
			if (i <= freq)
			{
				image[row * width + 2*index] = white_pixel;
				image[row * width + 2*index+1] = white_pixel;
			}
			else
			{
				image[row * width + 2*index].x = image[row * width + 2*index].x >> 1;
				image[row * width + 2*index+1].x = image[row * width + 2*index+1].x >> 1;
				
				

				image[row * width + 2*index].y = image[row * width + 2*index].y >> 1;
				image[row * width + 2*index+1].y = image[row * width + 2*index+1].y >> 1;
				


				image[row * width + 2*index].z = image[row * width + 2*index].z >> 1;
				image[row * width + 2*index+1].z = image[row * width + 2*index+1].z >> 1;
				
			}
		}
	
	}

}



__global__ void rgb2grayKernel(uchar4* image, uchar4* output,int width, int height)
{
	
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	int stride = blockDim.x * gridDim.x;
	
	for (int i = index; i < (width*height)-1; i+= stride) {
		
		unsigned char grey = grayScale(image[i].x, image[i].y, image[i].z);

		output[i].x = grey;
		output[i].y = grey;
		output[i].z = grey;
	}
}



__device__ unsigned char grayScale(unsigned char r, unsigned char g,unsigned char b)
{
	return 0.299*r + 0.587*g +  0.114*b;
}



int main( int argc, char** argv )
{
	// create input/output streams
	videoSource* input = videoSource::Create(argc, argv, ARG_POSITION(0));
	videoOutput* output = videoOutput::Create(argc, argv, ARG_POSITION(1));
	videoOutput* output2 = videoOutput::Create(argc, argv, ARG_POSITION(1));

	if ( !input )
	return 0;


	int hist_N = 256;
	int* hist_vector = (int *)malloc(hist_N*sizeof(int));	
	int* d_hist_vector = NULL;

	int N = 1280 * 720;
	uchar4* image_d = NULL;



	// Allocate device memory
	cudaMalloc(&image_d, N * sizeof(uchar4));
	cudaMalloc(&d_hist_vector, hist_N*sizeof(int));



	for(int i=0; i < hist_N; i++)
	{
		hist_vector[i] = 0;
	}

	cudaMemcpy(d_hist_vector, hist_vector, sizeof(int)*hist_N, cudaMemcpyHostToDevice);
	

	//get input
	uchar4* image = NULL; // can be uchar3, uchar4, float3, float4	
	int status = 0; // see videoSource::Status (OK, TIMEOUT, EOS, ERROR)	
	input->Capture(&image, 1000, &status);


	// Launch the kernel
	int blockSize = 256;
	int numBlocks = (N + blockSize - 1) / blockSize;
	for(int i = 0; i < blockSize; i++)
	{

		calcHistogramKernel<<<numBlocks, blockSize>>>(image, d_hist_vector, input->GetWidth(), input->GetHeight());
	}


	cudaMemcpy(hist_vector, d_hist_vector, sizeof(int)*hist_N, cudaMemcpyDeviceToHost);


	int added = 0;

	for(int i=0; i < 256; i++)
	{
		printf("Gray intensity %d: amount %d \n", i, hist_vector[i]);	
		added += hist_vector[i];
	}
	
	printf("Sum of pixels: %d\n", added);
		

	cudaFree(image_d);
}
