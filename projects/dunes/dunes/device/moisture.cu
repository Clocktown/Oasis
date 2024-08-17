#include "kernels.cuh"
#include "constants.cuh"
#include "grid.cuh"
#include <dunes/core/simulation_parameters.hpp>
#include <dunes/core/launch_parameters.hpp>
#include <sthe/device/vector_extension.cuh>

namespace dunes
{

	__global__ void evaporationKernel(Array2D<float4> terrainArray, Array2D<float> moistureArray) {
		const int2 cell{ getGlobalIndex2D() };

		if (isOutside(cell))
		{
			return;
		}

		float4 terrain{ terrainArray.read(cell) };
		float moisture{ moistureArray.read(cell) };

		terrain.w = terrain.w * exp(- c_parameters.evaporationRate * c_parameters.deltaTime);

		terrainArray.write(cell, terrain);
	}

	__global__ void moistureKernel(Array2D<float4> terrainArray, Array2D<float> moistureArray) {

	}

	__global__ void initMoistureDiffusionKernel(Buffer<float> moistureBuffer) {

	}

	__global__ void moistureDiffusionKernel(const Array2D<float4> terrainArray, const Array2D<float> moistureArray, Buffer<float> moistureBuffer) {

	}

	__global__ void finishMoistureDiffusionKernel(Array2D<float> moistureArray, const Buffer<float> moistureBuffer) {

	}

	void moisture(const LaunchParameters& t_launchParameters) {
		evaporationKernel << <t_launchParameters.gridSize2D, t_launchParameters.blockSize2D >> > (t_launchParameters.terrainArray, t_launchParameters.terrainMoistureArray);
	}
}