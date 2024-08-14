#include "kernels.cuh"
#include "constants.cuh"
#include "grid.cuh"
#include <dunes/core/simulation_parameters.hpp>
#include <dunes/core/launch_parameters.hpp>
#include <sthe/device/vector_extension.cuh>

namespace dunes
{

__global__ void pipeKernel(const Array2D<float4> t_terrainArray, Array2D<float4> t_waterFluxArray, const Array2D<float4> resistanceArray, const Array2D<float2> windArray)
{
	const int2 cell{ getGlobalIndex2D() };

	if (isOutside(cell))
	{
		return;
	}

	const float2 wind = /*(1.f - resistanceArray.read(cell).x) */ windArray.read(cell);
	const float windStrength = length(wind) + 1e-6f;
	const float2 windNorm = wind / windStrength;
	const float phase = 0.5f * (1.f + sin(c_parameters.wavePeriod * c_parameters.timestep * c_parameters.deltaTime));
	const float4 terrain{ t_terrainArray.read(cell) };
	const float sand{ terrain.x + terrain.y + terrain.z };
	const float water{ sand + terrain.w };
	float4 flux{ t_waterFluxArray.read(cell) };

	const float crossSectionalArea{ c_parameters.gridScale * c_parameters.gridScale };

	struct
	{
		int2 cell;
		float4 terrain;
		float sand;
		float water;
	} neighbor;

	for (int i{ 0 }; i < 4; ++i)
	{
		neighbor.cell = getWrappedCell(cell + c_offsets[i + i]);
		neighbor.terrain = t_terrainArray.read(neighbor.cell);
		neighbor.sand = neighbor.terrain.x + neighbor.terrain.y + neighbor.terrain.z;
		neighbor.water = neighbor.sand + neighbor.terrain.w;

		const float deltaHeight{ water - fmaxf(neighbor.water, sand) };

		*(&flux.x + i) = fmaxf(
			((1.f - 0.01f * c_parameters.deltaTime) * *(&flux.x + i)) + 
				c_parameters.deltaTime * crossSectionalArea  * c_parameters.rGridScale * 
					(deltaHeight + 
					c_parameters.waveStrength * (1.f - exp(-0.5f * c_parameters.waveDepthScale * c_parameters.waveDepthScale * (terrain.w + neighbor.terrain.w) * (terrain.w + neighbor.terrain.w))) * phase * windStrength * dot(windNorm, make_float2(c_offsets[i + i]))
					)
			, 0.0f);
	}

	flux *= fminf(terrain.w * c_parameters.gridScale * c_parameters.gridScale /
				  ((flux.x + flux.y + flux.z + flux.w) * c_parameters.deltaTime + 1e-06f), 1.0f);

	t_waterFluxArray.write(cell, flux);
}

__global__ void transportKernel(Array2D<float4> t_terrainArray, const Array2D<float4> t_waterFluxArray, Array2D<float2> t_waterVelocityArray)
{
	const int2 cell{ getGlobalIndex2D() };

	if (isOutside(cell))
	{
		return;
	}

	float4 terrain{ t_terrainArray.read(cell) };
	float4 flux{ t_waterFluxArray.read(cell) };

	struct
	{
		int2 cell;
		float4 flux;
	} neighbor;

	for (int i{ 0 }; i < 4; ++i)
	{
		neighbor.cell = getWrappedCell(cell + c_offsets[i + i]);
		neighbor.flux = t_waterFluxArray.read(neighbor.cell);

		*(&flux.x + i) -= *(&neighbor.flux.x + ((i + 2) % 4));
	}

	const float integrationScale{ c_parameters.rGridScale * c_parameters.rGridScale * c_parameters.deltaTime };
	terrain.w = fmaxf(terrain.w - integrationScale * (flux.x + flux.y + flux.z + flux.w), 0.0f);

	const float2 velocity{ 0.5f * float2{ flux.x - flux.y, flux.z - flux.w } };

	t_terrainArray.write(cell, terrain);
	t_waterVelocityArray.write(cell, velocity);
}

void transport(const LaunchParameters& t_launchParameters)
{
	pipeKernel<<<t_launchParameters.gridSize2D, t_launchParameters.blockSize2D>>>(t_launchParameters.terrainArray, t_launchParameters.fluxArray, t_launchParameters.resistanceArray, t_launchParameters.windArray);
	transportKernel<<<t_launchParameters.gridSize2D, t_launchParameters.blockSize2D>>>(t_launchParameters.terrainArray, t_launchParameters.fluxArray, t_launchParameters.waterVelocityArray);
}

}
