#include "kernels.cuh"
#include "constants.cuh"
#include "grid.cuh"
#include <dunes/core/simulation_parameters.hpp>
#include <dunes/core/launch_parameters.hpp>
#include <sthe/device/vector_extension.cuh>

namespace dunes
{

__global__ void venturiKernel(const Array2D<float4> t_terrainArray, Array2D<float2> t_windArray, const Array2D<float> moistureArray)
{
	const int2 index{ getGlobalIndex2D() };
	const int2 stride{ getGridStride2D() };

	int2 cell;

	for (cell.x = index.x; cell.x < c_parameters.gridSize.x; cell.x += stride.x)
	{
		for (cell.y = index.y; cell.y < c_parameters.gridSize.y; cell.y += stride.y)
		{

			const float4 terrain{ t_terrainArray.read(cell) };
			const float height{ terrain.x + terrain.y + terrain.z + terrain.w };

			const float venturiScale{ fmaxf(1.0f + c_parameters.venturiStrength * height, 0.5f) };
			const float2 windVelocity{ venturiScale * c_parameters.windSpeed * c_parameters.windDirection };

			//const float2 moistureGradient{
			//	0.5f * c_parameters.rGridScale * (moistureArray.read(getWrappedCell(cell + c_offsets[0])) - moistureArray.read(getWrappedCell(cell + c_offsets[4]))),
			//	0.5f * c_parameters.rGridScale * (moistureArray.read(getWrappedCell(cell + c_offsets[2])) - moistureArray.read(getWrappedCell(cell + c_offsets[6])))
			//};

			t_windArray.write(cell, windVelocity);// -100.f * moistureGradient);
		}
	}
}

void venturi(const LaunchParameters& t_launchParameters)
{
	venturiKernel<<<t_launchParameters.optimalGridSize2D, t_launchParameters.optimalBlockSize2D>>>(t_launchParameters.terrainArray, t_launchParameters.windArray, t_launchParameters.terrainMoistureArray);
}

}
