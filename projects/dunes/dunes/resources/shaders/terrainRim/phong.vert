#version 460 core

// Input
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

struct TerrainLayer
{
	vec3 diffuseColor;
    float specularIntensity;
	vec3 specularColor;
	float shininess;
	bool hasDiffuseMap;
	int pad1, pad2, pad3;
};

struct Terrain
{
	ivec2 gridSize;
	float gridScale;
	float heightScale;
	int tesselationLevel;
	int layerCount;
	bool hasHeightMap;
	bool hasAlphaMap;
	TerrainLayer layers[4];
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
	Terrain t_terrain;
};

// Output
out vec2 tescUV;

void main()
{
    const ivec2 subDivision = t_terrain.gridSize / t_terrain.tesselationLevel;
	const int numInstances = 2 * subDivision.x + 2 * subDivision.y;
	vec2 position = vec2(0);
	vec3 offset = vec3(0);
	if(gl_InstanceID < subDivision.x) {
		position = vec2(gl_InstanceID, 0);
		offset = vec3(gl_VertexID % 2, 0, gl_VertexID / 2);
	} else if (gl_InstanceID < 2 * subDivision.x) {
		position = vec2(gl_InstanceID - subDivision.x, subDivision.y);
		offset = vec3(1 - gl_VertexID % 2, 0, gl_VertexID / 2);
	} else if(gl_InstanceID < 2 * subDivision.x + subDivision.y) {
		position = vec2(0, gl_InstanceID - 2 * subDivision.x);
		offset = vec3(0, 1 - gl_VertexID % 2, gl_VertexID / 2);
	} else {
		position = vec2(subDivision.x, gl_InstanceID - (2 * subDivision.x + subDivision.y));
		offset = vec3(0, gl_VertexID % 2, gl_VertexID / 2);
	}

	tescUV = (position + offset.xy) / vec2(subDivision);

	gl_Position = vec4(t_terrain.gridScale * tescUV.x * float(t_terrain.gridSize.x), 
	                   offset.z, 
					   t_terrain.gridScale * tescUV.y * float(t_terrain.gridSize.y), 
					   1.0f);
}
