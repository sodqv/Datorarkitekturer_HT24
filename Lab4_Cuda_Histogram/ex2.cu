#include <jetson-utils/videoSource.h>
#include <jetson-utils/videoOutput.h>
#include <iostream>
#include <math.h>

__device__ char grayScale(char r, char g, char b);


__global__ void calcHistogramKernel(uchar4* image, int* hist_t, int width, int height)
{
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	int stride = blockDim.x * gridDim.x;

	for (int i = index; i < (width*height)-1; i+= stride) {
		
		char grey = grayScale(image[i].x, image[i].y, image[i].z);
		
		atomicAdd(&(hist_t[grey]), 1); 
		
	}
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
		
		char grey = grayScale(image[i].x, image[i].y, image[i].z);

		output[i].x = grey;
		output[i].y = grey;
		output[i].z = grey;
	}
}



__device__ char grayScale(char r, char g, char b)
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
	int N = 1280 * 720;

	int* hist_vector = (int *)malloc(hist_N*sizeof(int));	
	int* d_hist_vector = NULL;
	uchar4* image_d = NULL;
	int max_freq = 20000;


	
	
	// Allocate device memory	
	cudaMalloc(&image_d, N * sizeof(uchar4));
	cudaMalloc(&d_hist_vector, hist_N*sizeof(int));


		
	
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



		for(int i=0; i < hist_N; i++)
		{
			hist_vector[i] = 0;
		}
		cudaMemcpy(d_hist_vector, hist_vector, sizeof(int)*hist_N, cudaMemcpyHostToDevice);


		// Launch the kernel
		int blockSize = 256;
		int numBlocks = (N + blockSize - 1) / blockSize;	
		rgb2grayKernel<<<numBlocks, blockSize>>>(image, image_d, input->GetWidth(), input->GetHeight());
		calcHistogramKernel<<<numBlocks, blockSize>>>(image, d_hist_vector, input->GetWidth(), input->GetHeight());
		//cudaMemcpy(hist_vector, d_hist_vector, sizeof(int)*hist_N, cudaMemcpyDeviceToHost);
		
		int histNumBlocks = (hist_N + blockSize - 1) / blockSize;
		plotHistogramKernel<<<histNumBlocks, hist_N>>>(image_d, d_hist_vector, input->GetWidth(), input->GetHeight(), max_freq);


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


	cudaFree(image_d);
}
