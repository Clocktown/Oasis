#version 460 core

// Input
layout(location = 0) in vec4 t_position;

struct Light
{
	vec3 position;
	unsigned int type;
	vec3 color;
	float intensity;
	vec3 attenuation;
	float range;
	vec3 direction;
	float spotOuterCutOff;
	float spotInnerCutOff;
	int pad1, pad2, pad3;
};

struct Environment
{
	vec3 ambientColor;
	float ambientIntensity;
	vec3 fogColor;
	float fogDensity;
	unsigned int fogMode;
	float fogStart;
	float fogEnd;
	int lightCount;
	Light lights[16];
};

struct Material
{
	vec3 diffuseColor;
	float opacity;
	vec3 specularColor;
	float specularIntensity;
	float shininess;
	bool hasDiffuseMap;
	bool hasNormalMap;
	int pad;
};

layout(std140, binding = 0) uniform PipelineBuffer
{
	float t_time;
	float t_deltaTime;
	ivec2 t_resolution;
	mat4 t_projectionMatrix;
	mat4 t_viewMatrix;
	mat4 t_inverseViewMatrix;
	mat4 t_viewProjectionMatrix;
	Environment t_environment;
	mat4 t_modelMatrix;
	mat4 t_inverseModelMatrix;
	mat4 t_modelViewMatrix;
	mat4 t_inverseModelViewMatrix;
	ivec4 t_userID;
	Material t_material;
};

struct Vegetation
{
    vec3 pos; 
	int type;
	float health;
	float water;
	float age;
	float radius;
};

layout(std430, binding = 3) buffer VegBuffer
{
	Vegetation t_vegs[];
};

layout(std430, binding = 4) buffer VegMapBuffer
{
	int t_vegMap[];
};

// Functionality
void main()
{
    const Vegetation veg = t_vegs[t_vegMap[t_userID.x + gl_InstanceID]];
	gl_Position = t_viewProjectionMatrix * t_modelMatrix * vec4(veg.pos.xzy + veg.radius * t_position.xyz, 1.0f);
}
