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

struct Water
{
	ivec2 gridSize;
	float gridScale;
	float heightScale;
	int tesselationLevel;
	bool hasHeightMap;
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
	Water t_water;
};

layout(binding = 2) uniform sampler2D t_heightMap;

layout(quads, equal_spacing, ccw) in;
in vec2 teseUV[];

// Output
struct Fragment
{
	vec3 position;
	vec3 normal;
	vec2 uv;
};

out Fragment fragment;

vec2 getTerrainForNormal(ivec2 offset) {
	const vec4 t = textureOffset(t_heightMap, fragment.uv, offset);
	float sum = t.x + t.y + t.z;
	return vec2(sum, sum + t.w);
}

void main()
{
	const vec2 uv00 = teseUV[0];
	const vec2 uv10 = teseUV[1];
	const vec2 uv01 = teseUV[2];
	const vec2 uv11 = teseUV[3];
	const vec2 uv0 = mix(uv00, uv10, gl_TessCoord.y);
	const vec2 uv1 = mix(uv01, uv11, gl_TessCoord.y);
	fragment.uv = mix(uv0, uv1, gl_TessCoord.x);

    const vec2 position00 = gl_in[0].gl_Position.xz;
	const vec2 position10 = gl_in[1].gl_Position.xz;
	const vec2 position01 = gl_in[2].gl_Position.xz;
	const vec2 position11 = gl_in[3].gl_Position.xz;
	const vec2 position0 = mix(position00, position10, gl_TessCoord.y);
	const vec2 position1 = mix(position01, position11, gl_TessCoord.y);
	fragment.position.xz = mix(position0, position1, gl_TessCoord.x);

	if (t_water.hasHeightMap) 
	{
	    const vec4 terrain = texture(t_heightMap, fragment.uv).xyzw;
		const float height = t_water.heightScale * (terrain.x + terrain.y + terrain.z + terrain.w);
		fragment.position.y = height;

		const vec2 size = vec2(2.0f * t_water.gridScale,0.0f);
        const ivec3 offset = ivec3(-1, 0, 1);

		const vec2 terrain01 = getTerrainForNormal(offset.xy);
        const vec2 terrain21 = getTerrainForNormal(offset.zy);
        const vec2 terrain10 = getTerrainForNormal(offset.yx);
        const vec2 terrain12 = getTerrainForNormal(offset.yz);

		const vec3 wedge1 = normalize(vec3(size.x, t_water.heightScale * (terrain21.y - terrain01.y), size.y));
		const vec3 wedge2 = normalize(vec3(size.y, t_water.heightScale * (terrain12.y - terrain10.y), size.x));
		fragment.normal = cross(wedge2, wedge1);
	}
	else 
	{
	    fragment.position.y = 0.0f;
		fragment.normal = vec3(0.0f, 1.0f, 0.0f);
	}

	const vec4 position = t_modelMatrix * vec4(fragment.position, 1.0f);
	fragment.position = position.xyz;

    gl_Position = t_viewProjectionMatrix * position;
}