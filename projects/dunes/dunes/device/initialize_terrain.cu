#include "kernels.cuh"
#include "constants.cuh"
#include "grid.cuh"
#include <dunes/core/simulation_parameters.hpp>
#include <dunes/core/launch_parameters.hpp>
#include <sthe/device/vector_extension.cuh>

namespace dunes
{
#define M_PI 3.1415926535897932384626433832795

	// https://gist.github.com/patriciogonzalezvivo/670c22f3966e662d2f83
	__device__ float frand(float2 c) { return fract(sin(dot(c, float2{ 12.9898, 78.233 })) * 43758.5453); }
	__device__ float frand(float2 co, float l) { return frand(float2{ frand(co), l }); }
	__device__ float frand(float2 co, float l, float t) { return frand(float2{ frand(co, l), t }); }

	__device__ float perlin(float2 p, float dim, float time) {
		float3 pos = floorf(float3{ p.x * dim, p.y * dim, time });
		float3 posx = pos + float3{ 1.0f, 0.0f, 0.f };
		float3 posy = pos + float3{ 0.0f, 1.0f, 0.f };
		float3 posxy = pos + float3{ 1.0f, 1.0f, 0.f };
		float3 posz = pos + float3{ 0.f, 0.f, 1.f };
		float3 posxz = pos + float3{ 1.0f, 0.0f, 1.f };
		float3 posyz = pos + float3{ 0.0f, 1.0f, 1.f };
		float3 posxyz = pos + float3{ 1.0f, 1.0f, 1.f };
	
		float c = frand({ pos.x, pos.y }, dim, pos.z);
		float cx = frand({posx.x, posx.y}, dim, posx.z);
		float cy = frand({ posy.x, posy.y }, dim, posy.z);
		float cxy = frand({ posxy.x, posxy.y }, dim, posxy.z);

		float cz = frand({ posz.x, posz.y }, dim, posz.z);
		float cxz = frand({posxz.x, posxz.y}, dim, posxz.z);
		float cyz = frand({ posyz.x, posyz.y }, dim, posyz.z);
		float cxyz = frand({ posxyz.x, posxyz.y }, dim, posxyz.z);
	
		float3 d = fract(float3{ p.x * dim, p.y * dim, time });
		d = -0.5 * cos(d * M_PI) + 0.5;
	
		return lerp(bilerp(c, cx, cy, cxy, d.x, d.y), bilerp(cz, cxz, cyz, cxyz, d.x, d.y), d.z); // [0,1]
	}

	__device__ float seamless_perlin(float2 stretch, float2 uv, float2 border, float dim, float time)
	{
		float2 centered_uv_xy = 2.f * (uv - float2{ 0.5f, 0.5f });
		float2  limits = float2{ 1.f, 1.f } - 2.f * border;
		float2 distance = float2{ 0.f, 0.f };

		distance = max(abs(centered_uv_xy) - limits, float2{ 0.f, 0.f });

		centered_uv_xy = -1.f * sign(centered_uv_xy) * (limits + distance);

		distance /= 2.f * border;

		float2 xy_uv = 0.5f * centered_uv_xy + float2{ 0.5f, 0.5f };
		xy_uv = stretch * xy_uv;
		float2 base_uv = stretch * uv;

		float base_sample = perlin(base_uv, dim, time);

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

	__device__ float noise(float2 p, float freq)
	{
		float2 coords = p * freq;
		float2  ij = floorf(coords);
		float2  xy = coords - ij;

		//xy = 3.*xy*xy-2.*xy*xy*xy; // Alternative to cos
		xy = 0.5f * (1.f - cos(M_PI * xy));
		float a = frand((ij + float2{ 0., 0. }));
		float b = frand((ij + float2{ 1., 0. }));
		float c = frand((ij + float2{ 0., 1. }));
		float d = frand((ij + float2{ 1., 1. }));
		return bilerp(a, b, c, d, xy.x, xy.y);
	}

	__device__ float pNoise(float2 p, int res)
	{
		float persistance = .5;
		float n = 0.;
		float normK = 0.;
		float f = 4.;
		float amp = 1.;
		for (int i = 0; i <= res; i++)
		{
			n += amp * noise(p, f);
			f *= 2.;
			normK += amp;
			amp *= persistance;
		}
		float nf = n / normK;
		return nf * nf * nf * nf;
	}

	__device__ float seamless_pNoise(float2 off, float2 stretch, float2 uv, int res, float2 border)
	{
		float2 centered_uv_xy = 2.f * (uv - float2{ 0.5f, 0.5f });
		float2  limits = float2{ 1.f, 1.f } - 2.f * border;
		float2 distance = float2{ 0.f, 0.f };

		distance = max(abs(centered_uv_xy) - limits, float2{ 0.f, 0.f });

		centered_uv_xy = -1.f * sign(centered_uv_xy) * (limits + distance);

		distance /= 2.f * border;

		float2 xy_uv = 0.5f * centered_uv_xy + float2{ 0.5f, 0.5f };
		xy_uv = off + stretch * xy_uv;
		float2 base_uv = off + stretch * uv;

		float base_sample = pNoise(base_uv, res);

		if ((distance.x <= 0.f) && (distance.y <= 0.f))
		{
			return base_sample;
		}

		return bilerp(
			base_sample,
			pNoise(float2{ xy_uv.x, base_uv.y }, res),
			pNoise(float2{ base_uv.x, xy_uv.y }, res),
			pNoise(xy_uv, res),
			0.5f * smoothstep(0.f, 1.f, distance.x),
			0.5f * smoothstep(0.f, 1.f, distance.y)
		);
	}

	__global__ void initializeTerrainKernel(InitializationParameters t_initializationParameters)
	{
		const int2 cell{ getGlobalIndex2D() };

		if (isOutside(cell))
		{
			return;
		}

		const float2 uv = (make_float2(cell) + 0.5f) / make_float2(c_parameters.gridSize);

		const float4 curr_terrain = c_parameters.terrainArray.read(cell);
		const float4 curr_resistance = c_parameters.resistanceArray.read(cell);
		const float curr_moisture = c_parameters.moistureArray.read(cell);

		const int indices[7]{
			(int)NoiseGenerationTarget::Bedrock,
			(int)NoiseGenerationTarget::Sand,
			(int)NoiseGenerationTarget::Vegetation,
			(int)NoiseGenerationTarget::AbrasionResistance,
			(int)NoiseGenerationTarget::Soil,
			(int)NoiseGenerationTarget::Water,
			(int)NoiseGenerationTarget::Moisture
		};
		float values[7] = { curr_terrain.x, curr_terrain.y, curr_resistance.y, curr_resistance.z, curr_terrain.z, curr_terrain.w, curr_moisture };

		for (int i = 0; i < 7; ++i) {
			const auto& params = t_initializationParameters.noiseGenerationParameters[indices[i]];
			values[i] = params.enabled ?
				(params.bias +
					(params.uniform_random ? params.scale * frand(params.offset + params.stretch * uv) :
						(params.flat ?
							0.0f :
							params.scale * seamless_pNoise(params.offset, params.stretch, uv, params.iters, params.border)))) :
				values[i];
		}

		values[2] = clamp(values[2], 0.f, 1.f);
		values[3] = clamp(values[3], 0.f, 1.f);
		values[1] = fmaxf(values[1], 0.f);
		values[4] = fmaxf(values[4], 0.f);
		values[5] = fmaxf(values[5] - (values[0] + values[1] + values[4]), 0.f);
		values[6] = fmaxf(values[6], 0.f);

		// Regular initialization
		const float4 terrain{ values[0], values[1], values[4], values[5]};
		c_parameters.terrainArray.write(cell, terrain);

		const float4 resistance{ 0.0f, values[2], values[3], 0.0f};
		c_parameters.resistanceArray.write(cell, resistance);

		const int idx = getCellIndex(cell);
		c_parameters.slabBuffer[idx] = 0.0f;
		c_parameters.moistureArray.write(cell, values[6]);
		c_parameters.fluxArray.write(cell, { 0.f,0.f,0.f,0.f });
		c_parameters.velocityArray.write(cell, { 0.f, 0.f });
		c_parameters.sedimentArray.write(cell, 0.f);
	}

	// Rain Kernel here because we have noise functions defined here. The noise functions have to be compiled with fast math disabled to be accurate.
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

		float h = seamless_perlin(
			{ c_parameters.gridSize.x / fmaxf(c_parameters.gridSize.x, c_parameters.gridSize.y), c_parameters.gridSize.y / fmaxf(c_parameters.gridSize.x, c_parameters.gridSize.y) }, 
			uv, 
			{0.1f, 0.1f}, 
			rainScale, 
			rainPeriod * c_parameters.timestep * c_parameters.deltaTime
		);
		terrain.w += rainStrength * (h < rainProbability ? 1.f : 0.f);//clamp(h - (1.f - rainProbability), 0.f, 1.f);

		c_parameters.terrainArray.write(cell, terrain);
	}

	__global__ void addSandForCoverageKernel(float amount)
	{
		const int2 cell{ getGlobalIndex2D() };

		if (isOutside(cell))
		{
			return;
		}

		float4 curr_terrain = c_parameters.terrainArray.read(cell);

		curr_terrain.y += frand(make_float2(cell)) * 2.f * amount;
		curr_terrain.y = fmaxf(curr_terrain.y, 0.f);

		c_parameters.terrainArray.write(cell, curr_terrain);
	}

	__global__ void addSandCircleForCoverageKernel(int2 pos, int radius, float amount)
	{
		const int2 cell{ getGlobalIndex2D() };

		if (isOutside(cell))
		{
			return;
		}

		float4 curr_terrain = c_parameters.terrainArray.read(cell);

		const float2 cellf{ make_float2(cell) };
		const float2 posf{ make_float2(pos) };
		float2 distance{ cellf.x - posf.x, cellf.y - posf.y };
		distance.x = fminf(fabsf(distance.x), fminf(fabsf(distance.x + c_parameters.gridSize.x), fabsf(distance.x - c_parameters.gridSize.x)));
		distance.y = fminf(fabsf(distance.y), fminf(fabsf(distance.y + c_parameters.gridSize.y), fabsf(distance.y - c_parameters.gridSize.y)));

		if (length(distance) <= radius) {
			curr_terrain.y += frand(make_float2(cell)) * 2.f * amount;;
		}


		c_parameters.terrainArray.write(cell, curr_terrain);
	}

	void rain(const LaunchParameters& t_launchParameters) {
		rainKernel << <t_launchParameters.gridSize2D, t_launchParameters.blockSize2D >> > ();
	}

	void initializeTerrain(const LaunchParameters& t_launchParameters, const InitializationParameters& t_initializationParameters)
	{
		initializeTerrainKernel << <t_launchParameters.gridSize2D, t_launchParameters.blockSize2D >> > (t_initializationParameters);
	}

	void addSandForCoverage(const LaunchParameters& t_launchParameters, int2 res, bool uniform, int radius, float amount) {
		if (uniform) {
			addSandForCoverageKernel << <t_launchParameters.gridSize2D, t_launchParameters.blockSize2D >> > (amount);
		}
		else {
			addSandCircleForCoverageKernel << <t_launchParameters.gridSize2D, t_launchParameters.blockSize2D >> > (int2{ rand() % res.x, rand() % res.y }, radius, amount);
		}
	}

}
