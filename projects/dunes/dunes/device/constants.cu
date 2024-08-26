#include <dunes/core/simulation_parameters.hpp>
#include <cuda_runtime.h>

#define SQRT2 1.414213562f
#define RSQRT2 0.707106782f

namespace dunes
{

__constant__ SimulationParameters c_parameters{};

__constant__ int2 c_offsets[8]{ int2{ 1, 0 }, int2{ 1, 1 }, int2{ 0, 1 }, int2{ -1, 1 },
                                int2{ -1, 0 }, int2{ -1, -1 }, int2{ 0, -1 }, int2{ 1, -1 } };

__constant__ float c_distances[8]{ 1.0f, SQRT2, 1.0f, SQRT2, 1.0f, SQRT2, 1.0f, SQRT2 };
__constant__ float c_rDistances[8]{ 1.0f, RSQRT2, 1.0f, RSQRT2, 1.0f, RSQRT2, 1.0f, RSQRT2 };
// TODO: Dominance/compatibility Matrix
__constant__ float c_vegetationMatrix[2][2]{
    {1.f, 1.f},
    {1.f, 1.f}
};
__constant__ int c_numVegetationTypes{ 2 };
__constant__ VegetationType c_vegTypes[2]{
    {
        // Trees or something
        20.f,
        0.1f,
        100.f,
        0.2f,
        {2.f, 1.f},
        0.1f,
        1.f,
        0.1f,
        0.8f,
        1.f,
        0.2f,
        {0.75f, 0.25f},
        0.5f,
        0.001f,
        1.f,
        3.f,
        0.01f
    }, 
    {
        // Seaweed or something
        1.f,
        0.3f,
        30.f,
        0.1f,
        {3.f, 1.f},
        0.1f,
        1.f,
        1.0f,
        1.f,
        1.f,
        0.2f,
        {0.75f, 0.25f},
        0.25f,
        0.01f,
        1.f,
        0.1f,
        0.001f
    } 
};
}
