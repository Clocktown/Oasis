#version 460 core

// Input
struct Fragment
{
	vec2 uv;
};

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

struct RenderParameters
{
	vec4 sandColor;
	vec4 bedrockColor;
	vec4 windShadowColor;
	vec4 vegetationColor;
	vec4 soilColor;
	vec4 waterColor;
	vec4 humusColor;
	vec4 wetColor;
	float shadowStrength;
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

layout(std140, binding = 4) uniform RenderParametersBuffer
{
	RenderParameters renderParameters;
};

layout(binding = 2) uniform sampler2D t_heightMap;
layout(binding = 11) uniform sampler2D t_sedimentMap;
layout(binding = 12) uniform sampler2D t_shadowMap;
layout(binding = 13) uniform sampler2D t_vegHeightMap;

in Fragment fragment;

layout(binding = 0) uniform sampler2D tex; // .rgb = terrain/vegetation color, .a = sediment € [0,1]
layout(binding = 1) uniform sampler2D depthTex; // .r = terrain depth; cleared to 0
layout(binding = 4) uniform sampler2D waterTex; // .rgb = specular reflection color; .a = fresnel; cleared to 0
layout(binding = 3) uniform sampler2D waterDepthTex; // .r = water surface depth, .g = depth to AABB; cleared to 0. .ba not used right now, but intended to store screen space refraction offset if wanted

out vec4 fragCol;

void main() {
	vec4 terrainColor = texture(tex, fragment.uv).rgba;
	fragCol.rgb = terrainColor.rgb;

	const vec4 terrainDepth = texture(depthTex, fragment.uv);
	const vec2 waterDepth = texture(waterDepthTex, fragment.uv).rg;
	const vec2 terrainUV = ((1.f/t_water.gridSize) * (1.f/t_water.gridScale) * (t_inverseModelMatrix * vec4(terrainDepth.y, 0.f, terrainDepth.z, 1.f)).xz);
	const float sediment = 10.f * texture(t_sedimentMap, terrainUV).r;
	const float terrainHeight = terrainDepth.w;

	const vec2 shadow = texture(t_shadowMap, terrainUV).rg;
	const vec2 shadowHeights = texture(t_vegHeightMap, terrainUV).rg;
	const float terrainShadowHeight = (t_inverseModelMatrix * vec4(0, terrainHeight, 0, 1)).y;
	const float heightDiff = (shadowHeights.y - terrainShadowHeight);
	const float shadow_t = (shadowHeights.y - shadowHeights.x) > 1e-6f ? clamp(heightDiff / (shadowHeights.y - shadowHeights.x), 0, 1) : 1.f;
	const float shadowVal = mix(shadow.y, shadow.x, shadow_t);
	//const float shadowVal = terrainColor.a == 0.f ? 1.f : clamp(exp(0.1f * terrainShadowHeight) * shadow, 0, 1);
	if(terrainColor.a > 0.5f) {
		terrainColor *= pow(1.f - (renderParameters.shadowStrength * (1.f - shadowVal)), 2.2f);
	}

	const float waterHeight = (t_modelMatrix * vec4(0, dot(texture(t_heightMap, terrainUV).xyzw, vec4(1)), 0, 1)).y;
	const float t = max(waterHeight - terrainHeight, 0.f);
	const vec4 waterHighlight = texture(waterTex, fragment.uv);

	const float d = waterDepth.x > 0 ? (terrainDepth.x > 0 ? max(terrainDepth.x - waterDepth.x, 0.f) : max(waterDepth.y - waterDepth.x, 0.f)) : 0.f;
	const float viewDepthInterpolation = min(exp(-d), 1);

	const float topDepthInterpolation = min(exp(- (10.f * sediment + 1.f) * texture(tex, fragment.uv).a * t), 1);
	const vec3 waterSedimentColor = mix(renderParameters.waterColor.rgb, renderParameters.soilColor.rgb, min(sediment, 1.f));
	terrainColor.rgb = mix(mix(vec3(0), waterSedimentColor * terrainColor.rgb, topDepthInterpolation), terrainColor.rgb, topDepthInterpolation);
	terrainColor.rgb = mix(mix(0.2f * renderParameters.waterColor.rgb, renderParameters.waterColor.rgb, viewDepthInterpolation), terrainColor.rgb, viewDepthInterpolation);

	if(waterDepth.x > 0.f && renderParameters.waterColor.a > 0.5f) {
		fragCol.rgb = (1.f - waterHighlight.a) * terrainColor.rgb;//mix(mix(vec3(0,0,1), vec3(0.25,0.25,0), min(10.f * sediment, 1.f)), terrainColor.rgb, min(exp(-d), 1));
		fragCol.rgb += waterHighlight.a * waterHighlight.rgb;
	} else {
		fragCol.rgb = terrainColor.rgb;
	}
	//fragCol.rgb = vec3(shadowVal);

}