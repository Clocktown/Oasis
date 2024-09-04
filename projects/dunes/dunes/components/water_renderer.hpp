#pragma once

#include <sthe/core/component.hpp>
#include <sthe/core/terrain.hpp>
#include <sthe/core/custom_material.hpp>
#include <dunes/core/water.hpp>
#include <memory>
#include <vector>

namespace dunes
{
using namespace sthe;
class WaterRenderer : public Component
{
public:
	// Constructors
	explicit WaterRenderer(const std::shared_ptr<Water>& t_water = nullptr, const std::shared_ptr<CustomMaterial>& t_material = nullptr, const std::shared_ptr<CustomMaterial>& t_rimMaterial = nullptr);
	WaterRenderer(const WaterRenderer& t_waterRenderer) = delete;
	WaterRenderer(WaterRenderer&& t_waterRenderer) = default;

	// Destructor
	~WaterRenderer() = default;

	// Operators
	WaterRenderer& operator=(const WaterRenderer& t_waterRenderer) = delete;
	WaterRenderer& operator=(WaterRenderer&& t_waterRenderer) = default;

	// Setters
	void setWater(const std::shared_ptr<Water>& t_water);
	void setMaterial(const std::shared_ptr<CustomMaterial>& t_material);
	void setRimMaterial(const std::shared_ptr<CustomMaterial>& t_material);

	// Getters
	const std::shared_ptr<Water>& getWater() const;
	bool hasWater() const;
	const std::shared_ptr<CustomMaterial>& getMaterial() const;
	bool hasMaterial() const;
	const std::shared_ptr<CustomMaterial>& getRimMaterial() const;
	bool hasRimMaterial() const;
private:
	// Attribute 
	std::shared_ptr<Water> m_water;
	std::shared_ptr<CustomMaterial> m_material;
	std::shared_ptr<CustomMaterial> m_rimMaterial;
};

}
