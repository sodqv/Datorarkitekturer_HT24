#include <jetson-utils/videoSource.h>
#include <jetson-utils/videoOutput.h>
#include <iostream>
#include <math.h>

__device__ unsigned char grayScale(unsigned char r, unsigned char g, unsigned char b);


__global__ void calcHistogramKernel(uchar4* image, int* hist_t, int width, int height)
{


	int index = blockIdx.x * blockDim.x + threadIdx.x;
	int stride = blockDim.x * gridDim.x;

	for (int i = index; i < (width*height)-1; i+= stride) {
		
//		unsigned char grey = grayScale(image[i].x, image[i].y, image[i].z);
		unsigned char grey = image[i].x;
		
		atomicAdd(&(hist_t[grey]), 1); 
		
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



__device__ unsigned char grayScale(unsigned char r, unsigned char g, unsigned char b)
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
		cudaMemcpy(d_hist_vector, hist_vector, sizeof(int)*hist_N, cudaMemcpyHostToDevice);
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
		
	/*
	// capture/display loop
	while (true)
	{

		uchar4* image = NULL; // can be uchar3, uchar4, float3, float4	

		int status = 0; // see videoSource::Status (OK, TIMEOUT, EOS, ERROR)

		if ( !input->Capture(&image, 1000, &status) ) // 1000ms timeout (default)
		{
			if (status == videoSource::TIMEOUT)
				continue;
		
			break; // EOS
		}



		// Launch the kernel
		int blockSize = 256;
		int numBlocks = (N + blockSize - 1) / blockSize;	
		rgb2grayKernel<<<numBlocks, blockSize>>>(image, image_d, input->GetWidth(), input->GetHeight());


		//Camera window 1
		if ( output != NULL )
		{
			output->Render(image, input->GetWidth(), input->GetHeight());

			// Update status bar
			char str[256];
		
			sprintf(str, "Camera Viewer (%ux%u) | %0.1f FPS", input->GetWidth(),

			input->GetHeight(), output->GetFrameRate());
			output->SetStatus(str);

		if (!output->IsStreaming()) // check if the user quit
		break;
		}

		//Camera window 2
		if ( output2 != NULL )
		{
			output2->Render(image_d, input->GetWidth(), input->GetHeight());

			// Update status bar
			char str[256];
		
			sprintf(str, "Camera Viewer (%ux%u) | %0.1f FPS", input->GetWidth(),

			input->GetHeight(), output2->GetFrameRate());
			output2->SetStatus(str);

		if (!output2->IsStreaming()) // check if the user quit
		break;
		}
	}
*/

	cudaFree(image_d);
}
