#include "kernels.cuh"
#include "constants.cuh"
#include "grid.cuh"
#include "common.cuh"
#include <dunes/core/simulation_parameters.hpp>
#include <dunes/core/launch_parameters.hpp>
#include <sthe/device/vector_extension.cuh>

namespace dunes
{

__global__ void pipeKernel(const Array2D<float4> t_terrainArray, Array2D<float4> t_waterFluxArray, const Array2D<float4> resistanceArray, const Array2D<float2> windArray, Buffer<float> slopeBuffer)
{
	const int2 cell{ getGlobalIndex2D() };

	if (isOutside(cell))
	{
		return;
	}

	const float2 wind = /*(1.f - resistanceArray.read(cell).x) */ windArray.read(cell);
	const float windStrength = length(wind) + 1e-6f;
	const float2 windNorm = wind / windStrength;
	const float phase = fmaxf(sin(c_parameters.wavePeriod * c_parameters.timestep * c_parameters.deltaTime), 0.f);
	const float4 terrain{ t_terrainArray.read(cell) };
	const float sand{ terrain.x + terrain.y + terrain.z };
	const float water{ sand + terrain.w };
	float4 flux{ t_waterFluxArray.read(cell) };

	const float crossSectionalArea{ c_parameters.gridScale * c_parameters.gridScale };

	float heights[4];

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
		heights[i] = neighbor.sand;

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

	const float3 gX{ 2.f * c_parameters.gridScale, heights[0] - heights[2], 0.f};
	const float3 gY{ 0.f, heights[1] - heights[3], 2.f * c_parameters.gridScale};
	const float cos_alpha{ normalize(cross(gY, gX)).y };
	const float min_sin_alpha = 0.5f;
	const float sin_alpha = clamp(min_sin_alpha + (1.f - min_sin_alpha) * sqrt(1.f - cos_alpha * cos_alpha), 0.f, 1.f);
	slopeBuffer[getCellIndex(cell)] = sin_alpha;

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

	//TODO: make optional, also not working properly, may need to add it to different kernels.
	setBorderWaterLevelMin(cell, terrain, c_parameters.waterBorderLevel);
	setWaterLevelMin(cell, terrain, c_parameters.waterLevel);

	t_terrainArray.write(cell, terrain);
	t_waterVelocityArray.write(cell, velocity);
}

__global__ void initSedimentKernel(Buffer<float> advectedSedimentBuffer) {
	const int2 cell{ getGlobalIndex2D() };

	if (isOutside(cell))
	{
		return;
	}
	const int idx = getCellIndex(cell);

	advectedSedimentBuffer[idx] = 0.f;
}

__global__ void sedimentExchangeKernel(Array2D<float> sedimentArray, const Array2D<float2> velocityArray, const Buffer<float> advectedSedimentBuffer, const Buffer<float> slopeBuffer, Array2D<float4> terrainArray, Array2D<float4> resistanceArray, const Array2D<float> moistureArray) {
	const int2 cell{ getGlobalIndex2D() };

	if (isOutside(cell))
	{
		return;
	}
	const int idx = getCellIndex(cell);

	const float slope = slopeBuffer[idx];
	float4 resistance = resistanceArray.read(cell);

	float sediment = advectedSedimentBuffer[idx];
	float4 terrain = terrainArray.read(cell);
	const float moisture = moistureArray.read(cell);
	// Humus conversion TODO: UI parameter
	const float humusConversion = fminf(0.01f * c_parameters.deltaTime * (0.01f + 0.99f * resistance.y) * moisture, fminf(terrain.y, resistance.w));
	terrain.y -= humusConversion;
	terrain.z += humusConversion;
	resistance.w -= humusConversion;
	// Soil drying out TODO: UI parameter
	const float drying = fminf(0.01f * c_parameters.deltaTime * (1.f - 0.99f * resistance.y) * fmaxf(1.f - 50.f * moisture, 0.f) * expf(-10.f * terrain.y), terrain.z);
	resistance.w -= fminf(drying, resistance.w);
	terrain.z -= drying;
	terrain.y += drying;


	const float speed = length(velocityArray.read(cell)) / fmaxf(0.1f * terrain.w, 1.f); // Velocity is at the surface, so decrease it for deep water

	const float sedimentCapacity = c_parameters.sedimentCapacityConstant * slope * speed * (1.f - 0.5f * resistance.y) * fminf(0.1f + 0.9f * terrain.w, 1.f);
	const float sandDissolutionRate = c_parameters.sandDissolutionConstant * c_parameters.deltaTime;
	const float soilDissolutionRate = c_parameters.soilDissolutionConstant * c_parameters.deltaTime;
	const float bedrockDissolutionRate = c_parameters.bedrockDissolutionConstant * c_parameters.deltaTime;
	const float depositionRate = c_parameters.sedimentDepositionConstant * (1 + resistance.y) * c_parameters.deltaTime;
	if (sediment <= sedimentCapacity) {
		const float sandDissolution = fminf((sedimentCapacity - sediment) * sandDissolutionRate, terrain.y);
		sediment += sandDissolution;
		terrain.y -= sandDissolution;
	}
	if (terrain.y <= 0.f && sediment <= sedimentCapacity) {
		const float soilDissolution = fminf((sedimentCapacity - sediment) * soilDissolutionRate, terrain.z);
		sediment += soilDissolution;
		terrain.z -= soilDissolution;
	}
	if (terrain.z <= 0.f && sediment <= sedimentCapacity) {
		const float bedrockDissolution = (sedimentCapacity - sediment) * bedrockDissolutionRate;
		sediment += bedrockDissolution;
		terrain.x -= bedrockDissolution;
	}
	if (sediment > sedimentCapacity) {
		const float sandDeposition = (sediment - sedimentCapacity) * depositionRate;
		sediment -= sandDeposition;
		terrain.y += sandDeposition;
	}

	sedimentArray.write(cell, sediment);
	terrainArray.write(cell, terrain);
	resistanceArray.write(cell, resistance);
}

__global__ void sedimentAdvectionKernel(const Array2D<float> sedimentArray, const Array2D<float2> velocityArray, Buffer<float> advectedSedimentBuffer) {
	// Backward
	/*const int2 cell{getGlobalIndex2D()};

	if (isOutside(cell))
	{
		return;
	}

	const float2 pos{ make_float2(cell) + 0.5f -
		c_parameters.deltaTime * c_parameters.rGridScale * velocityArray.read(cell) 
	};

	advectedSedimentBuffer[getCellIndex(cell)] = sedimentArray.sample(pos);*/
	// Forward

	const int2 cell{ getGlobalIndex2D() };

	if (isOutside(cell))
	{
		return;
	}

	const float slab{ sedimentArray.read(cell) };

	const float2 velocity{ velocityArray.read(cell) };

	const float2 position{ make_float2(cell) };

	if (slab > 0.0f)
	{
		const float2 nextPosition{ position + velocity * c_parameters.rGridScale * c_parameters.deltaTime };

		const int2 nextCell{ make_int2(floorf(nextPosition)) };

		for (int x{ nextCell.x }; x <= nextCell.x + 1; ++x)
		{
			const float u{ 1.0f - abs(static_cast<float>(x) - nextPosition.x) };

			for (int y{ nextCell.y }; y <= nextCell.y + 1; ++y)
			{
				const float v{ 1.0f - abs(static_cast<float>(y) - nextPosition.y) };
				const float weight{ u * v };

				if (weight > 0.0f)
				{
					atomicAdd(advectedSedimentBuffer + getCellIndex(getWrappedCell(int2{ x,y })), weight * slab);
				}
			}
		}
	}
}

void transport(const LaunchParameters& t_launchParameters, const SimulationParameters& t_simulationParameters)
{
	Buffer<float> slopeBuffer{ t_launchParameters.tmpBuffer + t_simulationParameters.cellCount };

	pipeKernel<<<t_launchParameters.gridSize2D, t_launchParameters.blockSize2D>>>(t_launchParameters.terrainArray, t_launchParameters.fluxArray, t_launchParameters.resistanceArray, t_launchParameters.windArray, slopeBuffer);
	transportKernel<<<t_launchParameters.gridSize2D, t_launchParameters.blockSize2D>>>(t_launchParameters.terrainArray, t_launchParameters.fluxArray, t_launchParameters.waterVelocityArray);
}

void sediment(const LaunchParameters& t_launchParameters, const SimulationParameters& t_simulationParameters) {
	Buffer<float> advectedSedimentBuffer{ t_launchParameters.tmpBuffer };
	Buffer<float> slopeBuffer{ t_launchParameters.tmpBuffer + t_simulationParameters.cellCount };

	initSedimentKernel << <t_launchParameters.gridSize2D, t_launchParameters.blockSize2D >> > (advectedSedimentBuffer);
	sedimentAdvectionKernel<<<t_launchParameters.gridSize2D, t_launchParameters.blockSize2D>>>(t_launchParameters.sedimentArray, t_launchParameters.waterVelocityArray, advectedSedimentBuffer);
	sedimentExchangeKernel<<<t_launchParameters.gridSize2D, t_launchParameters.blockSize2D>>>(t_launchParameters.sedimentArray, t_launchParameters.waterVelocityArray, advectedSedimentBuffer, slopeBuffer, t_launchParameters.terrainArray, t_launchParameters.resistanceArray, t_launchParameters.terrainMoistureArray);
}

}
