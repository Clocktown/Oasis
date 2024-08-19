#pragma once

#include "constants.cuh"
#include "grid.cuh"

namespace dunes {
	__forceinline__ __device__ void setBorderWaterLevelMin(int2 cell, float4& terrain, float level) {
		terrain.w = isBorder(cell) ? fmaxf(terrain.w, fmaxf(level - (terrain.x + terrain.y + terrain.z), 0.f)) : terrain.w;
	}

	__forceinline__ __device__ void setBorderWaterLevel(int2 cell, float4& terrain, float level) {
		terrain.w = isBorder(cell) ? fmaxf(level - (terrain.x + terrain.y + terrain.z), 0.f) : terrain.w;
	}
}