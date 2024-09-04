#pragma once

#include <sthe/core/custom_material.hpp>
#include <sthe/gl/vertex_array.hpp>
#include <sthe/gl/buffer.hpp>
#include <sthe/gl/texture2d.hpp>
#include <memory>
#include <array>
#include <vector>

namespace dunes
{

namespace uniform
{

struct Water
{
	glm::ivec2 gridSize;
	float gridScale;
	float heightScale;
	int tesselationLevel;
	int hasHeightMap;
};

}

using namespace sthe;
class Water
{
public:
	// Constructors
	explicit Water(const glm::ivec2& t_gridSize = glm::ivec2{ 512 }, const float t_gridScale = 1.0f, const float t_heightScale = 1.0f);
	Water(const Water& t_water) = default;
	Water(Water&& t_water) = default;

	// Destructor
	~Water() = default;

	// Operators
	Water& operator=(const Water& t_terrain) = default;
	Water& operator=(Water&& t_terrain) = default;

	// Functionality
	void bind() const;

	// Setters
	void setGridSize(const glm::ivec2& t_gridSize);
	void setGridScale(const float t_gridScale);
	void setHeightScale(const float t_heightScale);
	void setTesselationLevel(const int t_tesselationLevel);
	void setHeightMap(const std::shared_ptr<gl::Texture2D>& t_heightMap);
	
	// Getters
	const glm::ivec2& getGridSize() const;
	const float getGridScale() const;
	const float getHeightScale() const;
	int getTesselationLevel() const;
	const std::shared_ptr<gl::VertexArray> getVertexArray() const;
	const std::shared_ptr<gl::Texture2D>& getHeightMap() const;

	bool hasHeightMap() const;

private:
	// Attributes
	glm::ivec2 m_gridSize;
	float m_gridScale;
	float m_heightScale;
	int m_tesselationLevel;
	std::shared_ptr<gl::VertexArray> m_vertexArray;
	std::shared_ptr<gl::Texture2D> m_heightMap;
};

}
