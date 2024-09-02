#pragma once

#include <sthe/core/component.hpp>
#include <glm/glm.hpp>

namespace sthe
{

class Renderer : public Component
{
public:
	// Constructors
	Renderer();
	Renderer(const Renderer& t_renderer) = delete;
	Renderer(Renderer&& t_renderer) = default;

	// Destructor
	virtual ~Renderer() = 0;

	// Operators
	Renderer& operator=(const Renderer& t_renderer) = delete;
	Renderer& operator=(Renderer&& t_renderer) = default;

	// Setters
	void setBaseInstance(const int t_baseInstance);
	void setInstanceCount(const int t_instanceCount);
	void setUserID(const glm::ivec4& userID);

	// Getters
	int getBaseInstance() const;
	int getInstanceCount() const;
	const glm::ivec4& getUserID() const;
private:
	// Attributes
	int m_baseInstance;
	int m_instanceCount;
	glm::ivec4 m_userID;
};

}
