#include "water.hpp"
#include <sthe/config/config.hpp>
#include <sthe/config/binding.hpp>
#include <sthe/core/custom_material.hpp>
#include <sthe/gl/vertex_array.hpp>
#include <sthe/gl/buffer.hpp>
#include <sthe/gl/texture2d.hpp>
#include <memory>
#include <vector>

namespace dunes
{
using namespace sthe;
// Constructor
Water::Water(const glm::ivec2& t_gridSize, const float t_gridScale, const float t_heightScale) :
	m_gridSize{ t_gridSize },
	m_gridScale{ t_gridScale },
	m_heightScale{ t_heightScale },
	m_tesselationLevel{ 32 },
	m_vertexArray{ std::make_shared<gl::VertexArray>() },
	m_heightMap{ nullptr }
{
	STHE_ASSERT(t_gridSize.x > 0, "Grid size x must be greater than 0");
	STHE_ASSERT(t_gridSize.y > 0, "Grid size y must be greater than 0");
	STHE_ASSERT(t_gridScale != 0.0f, "Grid scale cannot be equal to 0");
	STHE_ASSERT(t_heightScale != 0.0f, "Height scale cannot be equal to 0");

}

// Functionality
void Water::bind() const
{
	m_vertexArray->bind();

	if (hasHeightMap())
	{
		m_heightMap->bind(STHE_TEXTURE_UNIT_TERRAIN_HEIGHT);
	}
}

// Setters
void Water::setGridSize(const glm::ivec2& t_gridSize)
{
	STHE_ASSERT(t_gridSize.x > 0, "Grid size x must be greater than 0");
	STHE_ASSERT(t_gridSize.y > 0, "Grid size y must be greater than 0");

	m_gridSize = t_gridSize;
}

void Water::setGridScale(const float t_gridScale)
{
	STHE_ASSERT(t_gridScale != 0.0f, "Grid scale cannot be equal to 0");

	m_gridScale = t_gridScale;
}

void Water::setHeightScale(const float t_heightScale)
{
	STHE_ASSERT(t_heightScale != 0.0f, "Height scale cannot be equal to 0");

	m_heightScale = t_heightScale;
}

void Water::setTesselationLevel(const int t_tesselationLevel)
{
	STHE_ASSERT(t_tesselationLevel > 0, "Tesselation level must be greater than 0");

	m_tesselationLevel = t_tesselationLevel;
}

void Water::setHeightMap(const std::shared_ptr<sthe::gl::Texture2D>& t_heightMap)
{
	m_heightMap = t_heightMap;
}

// Getters
const glm::ivec2& Water::getGridSize() const
{
	return m_gridSize;
}

const float Water::getGridScale() const
{
	return m_gridScale;
}

const float Water::getHeightScale() const
{
	return m_heightScale;
}

int Water::getTesselationLevel() const
{
	return m_tesselationLevel;
}

const std::shared_ptr<gl::VertexArray> Water::getVertexArray() const
{
	return m_vertexArray;
}

const std::shared_ptr<gl::Texture2D>& Water::getHeightMap() const
{
	return m_heightMap;
}

bool Water::hasHeightMap() const
{
	return m_heightMap != nullptr;
}

}
