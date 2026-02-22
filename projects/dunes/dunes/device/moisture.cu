#include "kernels.cuh"
#include "constants.cuh"
#include "grid.cuh"
#include "common.cuh"
#include <dunes/core/simulation_parameters.hpp>
#include <dunes/core/launch_parameters.hpp>
#include <sthe/device/vector_extension.cuh>

namespace dunes
{

	__global__ void evaporationKernel() {
		const int2 cell{ getGlobalIndex2D() };

		if (isOutside(cell))
		{
			return;
		}

		float4 terrain{ c_parameters.terrainArray.read(cell) };
		float moisture{ c_parameters.moistureArray.read(cell) };

		const float sandMoistureRate = c_parameters.sandMoistureRate;
		const float soilMoistureRate = c_parameters.soilMoistureRate;
		const float iTerrainThicknessThreshold = c_parameters.iTerrainThicknessMoistureThreshold;
		const float sandFactor = 1.f - clamp(terrain.y * iTerrainThicknessThreshold, 0.f, 1.f);
		const float moistureRate = lerp(sandMoistureRate, soilMoistureRate, sandFactor);

		const float vegetation = c_parameters.resistanceArray.read(cell).y;
		const float moistureEvaporationFactor = (1.f - clamp(terrain.w * 10.f, 0.f, 1.f)) * (1.f - 0.75f * clamp(vegetation, 0.f, 1.f));
		terrain.w = terrain.w * exp(- c_parameters.evaporationRate * c_parameters.deltaTime);
		moisture = moisture * exp(-c_parameters.moistureEvaporationScale * moistureRate * moistureEvaporationFactor * c_parameters.deltaTime);

		setBorderWaterLevelMin(cell, terrain, c_parameters.waterBorderLevel);
		setWaterLevelMin(cell, terrain, c_parameters.waterLevel);

		c_parameters.terrainArray.write(cell, terrain);
		c_parameters.moistureArray.write(cell, moisture);
	}

	__global__ void moistureKernel() {
		const int2 cell{ getGlobalIndex2D() };

		if (isOutside(cell))
		{
			return;
		}

		float4 terrain{ c_parameters.terrainArray.read(cell) };
		float moisture{ c_parameters.moistureArray.read(cell) };
		const float sandMoistureRate = c_parameters.sandMoistureRate;
		const float soilMoistureRate = c_parameters.soilMoistureRate;
		const float iTerrainThicknessThreshold = c_parameters.iTerrainThicknessMoistureThreshold;
		const float terrainThickness = terrain.y + terrain.z;
		const float moistureCapacityConstant = c_parameters.moistureCapacityConstant;
		const float moistureCapacity = moistureCapacityConstant * clamp(terrainThickness * iTerrainThicknessThreshold, 0.f, 1.f);
		const float sandFactor = 1.f - clamp(terrain.y * iTerrainThicknessThreshold, 0.f, 1.f);
		const float moistureRate = moisture > (0.5f * moistureCapacity) ? 0.02f * lerp(sandMoistureRate, soilMoistureRate, sandFactor) : lerp(sandMoistureRate, soilMoistureRate, sandFactor);

		if (moisture > moistureCapacity) {
			terrain.w += moisture - moistureCapacity;
			moisture = moistureCapacity;
		}
		else {
			const float dMoisture = fminf(moistureRate * (moistureCapacity - moisture), terrain.w);
			terrain.w -= dMoisture;
			moisture += dMoisture;
		}

		setBorderWaterLevelMin(cell, terrain, c_parameters.waterBorderLevel);
		setWaterLevelMin(cell, terrain, c_parameters.waterLevel);

		c_parameters.terrainArray.write(cell, terrain);
		c_parameters.moistureArray.write(cell, moisture);
	}

	__global__ void initMoistureDiffusionKernel(Buffer<float> moistureBuffer) {
		const int2 cell{ getGlobalIndex2D() };

		if (isOutside(cell))
		{
			return;
		}

		moistureBuffer[getCellIndex(cell)] = 0.f;
	}

	__global__ void moistureDiffusionKernel(Buffer<float> moistureBuffer) {
		const int2 cell{ getGlobalIndex2D() };

		if (isOutside(cell))
		{
			return;
		}

		const float prev = c_parameters.moistureArray.read(cell);
		const float diffusion_coefficient = fminf(0.5f * c_parameters.rGridScale * c_parameters.rGridScale * c_parameters.deltaTime, 0.25f);

		float next = 0.f;
		for(int i = 0; i < 4; ++i) {
			const int2 nextCell = getWrappedCell(cell + c_offsets[2 * i]);

			next += c_parameters.moistureArray.read(nextCell);
		}
		next = prev + diffusion_coefficient * (next - 4 * prev); 

		moistureBuffer[getCellIndex(cell)] = next;
	}

	__global__ void finishMoistureDiffusionKernel(const Buffer<float> moistureBuffer) {
		const int2 cell{ getGlobalIndex2D() };

		if (isOutside(cell))
		{
			return;
		}

		c_parameters.moistureArray.write(cell, moistureBuffer[getCellIndex(cell)]);
	}

	// https://gist.github.com/patriciogonzalezvivo/670c22f3966e662d2f83
	__device__ __forceinline__ float frando(float2 c) { return fract(sin(dot(c, float2{ 12.9898f, 78.233f })) * 43758.5453f); }
	__device__ __forceinline__ float frand(float2 co, float l) { return frando(float2{ frando(co), l }); }
	__device__ __forceinline__ float frand(float2 co, float l, float t) { return frando(float2{ frand(co, l), t }); }

	#define M_PI 3.1415926535897932384626433832795f
	__device__ __forceinline__ float perlin(float2 p, float dim, float time) {
		const float3 pos = floorf(float3{ p.x * dim, p.y * dim, time });
		const float3 posx = pos + float3{ 1.0f, 0.0f, 0.f };
		const float3 posy = pos + float3{ 0.0f, 1.0f, 0.f };
		const float3 posxy = pos + float3{ 1.0f, 1.0f, 0.f };
		const float3 posz = pos + float3{ 0.f, 0.f, 1.f };
		const float3 posxz = pos + float3{ 1.0f, 0.0f, 1.f };
		const float3 posyz = pos + float3{ 0.0f, 1.0f, 1.f };
		const float3 posxyz = pos + float3{ 1.0f, 1.0f, 1.f };
	
		const float c = frand({ pos.x, pos.y }, dim, pos.z);
		const float cx = frand({posx.x, posx.y}, dim, posx.z);
		const float cy = frand({ posy.x, posy.y }, dim, posy.z);
		const float cxy = frand({ posxy.x, posxy.y }, dim, posxy.z);

		const float cz = frand({ posz.x, posz.y }, dim, posz.z);
		const float cxz = frand({posxz.x, posxz.y}, dim, posxz.z);
		const float cyz = frand({ posyz.x, posyz.y }, dim, posyz.z);
		const float cxyz = frand({ posxyz.x, posxyz.y }, dim, posxyz.z);
	
		float3 d = fract(float3{ p.x * dim, p.y * dim, time });
		d = -0.5f * cos(d * M_PI) + 0.5f;
	
		return lerp(bilerp(c, cx, cy, cxy, d.x, d.y), bilerp(cz, cxz, cyz, cxyz, d.x, d.y), d.z); // [0,1]
	}

	__device__ __forceinline__ float seamless_perlin(float2 stretch, float2 uv, float2 border, float dim, float time)
	{
		float2 centered_uv_xy = 2.f * (uv - float2{ 0.5f, 0.5f });
		const float2  limits = float2{ 1.f, 1.f } - 2.f * border;
		float2 distance = float2{ 0.f, 0.f };

		distance = max(abs(centered_uv_xy) - limits, float2{ 0.f, 0.f });

		centered_uv_xy = -1.f * sign(centered_uv_xy) * (limits + distance);

		distance /= 2.f * border;

		float2 xy_uv = 0.5f * centered_uv_xy + float2{ 0.5f, 0.5f };
		xy_uv = stretch * xy_uv;
		const float2 base_uv = stretch * uv;

		const float base_sample = perlin(base_uv, dim, time);

		if ((distance.x <= 0.f) && (distance.y <= 0.f))
		{
			return base_sample;
		}

		return bilerp(
			base_sample,
			perlin(float2{ xy_uv.x, base_uv.y }, dim, time),
			perlin(float2{ base_uv.x, xy_uv.y }, dim, time),
			perlin(xy_uv, dim, time),
			0.5f * smoothstep(0.f, 1.f, distance.x),
			0.5f * smoothstep(0.f, 1.f, distance.y)
		);
	}

	__global__ void rainKernel() {
		const int2 cell{ getGlobalIndex2D() };

		if (isOutside(cell))
		{
			return;
		}

		const float rainStrength = c_parameters.rainStrength;
		const float rainPeriod = c_parameters.rainPeriod;
		const float rainScale = c_parameters.rainScale; // Larger = finer resolution
		const float rainProbabilityMin = c_parameters.rainProbabilityMin;
		const float rainProbabilityMax = c_parameters.rainProbabilityMax;
		const float iRainProbabilityHeightRange = c_parameters.iRainProbabilityHeightRange;
		float4 terrain{ c_parameters.terrainArray.read(cell) };
		const float height = terrain.x + terrain.y + terrain.z;
		const float rainProbability = rainProbabilityMin + (rainProbabilityMax - rainProbabilityMin) * clamp(height * iRainProbabilityHeightRange, 0.f, 1.f);
		const float2 uv = (make_float2(cell) + 0.5f) / make_float2(c_parameters.gridSize);

		const float h = seamless_perlin(
			{ c_parameters.gridSize.x / fmaxf(c_parameters.gridSize.x, c_parameters.gridSize.y), c_parameters.gridSize.y / fmaxf(c_parameters.gridSize.x, c_parameters.gridSize.y) }, 
			uv, 
			{0.1f, 0.1f}, 
			rainScale, 
			rainPeriod * c_parameters.timestep * c_parameters.deltaTime
		);
		terrain.w += rainStrength * (h < rainProbability ? 1.f : 0.f);//clamp(h - (1.f - rainProbability), 0.f, 1.f);

		c_parameters.terrainArray.write(cell, terrain);
	}

	void rain(const LaunchParameters& t_launchParameters) {
		rainKernel << <t_launchParameters.gridSize2D, t_launchParameters.blockSize2D >> > ();
	}

	void moisture(const LaunchParameters& t_launchParameters, const SimulationParameters& t_simulationParameters) {
		Buffer<float> diffusedMoistureBuffer{ t_launchParameters.tmpBuffer };

		evaporationKernel << <t_launchParameters.gridSize2D, t_launchParameters.blockSize2D >> > ();
		moistureKernel << <t_launchParameters.gridSize2D, t_launchParameters.blockSize2D >> > ();

		initMoistureDiffusionKernel << <t_launchParameters.gridSize2D, t_launchParameters.blockSize2D >> > (diffusedMoistureBuffer);
		moistureDiffusionKernel << <t_launchParameters.gridSize2D, t_launchParameters.blockSize2D >> > (diffusedMoistureBuffer);
		finishMoistureDiffusionKernel << <t_launchParameters.gridSize2D, t_launchParameters.blockSize2D >> > (diffusedMoistureBuffer);
	}
}