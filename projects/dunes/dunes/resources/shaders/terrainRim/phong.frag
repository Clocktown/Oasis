#version 460 core
layout(early_fragment_tests) in;
// Constants
const int FOG_MODE_NONE = 0;
const int FOG_MODE_LINEAR = 1;
const int FOG_MODE_EXPONENTIAL = 2;
const int FOG_MODE_EXPONENTIAL2 = 3;
const int LIGHT_TYPE_SPOT = 1;
const int LIGHT_TYPE_DIRECTIONAL = 2;
const float EPSILON = 1e-6f;

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

struct Fragment
{
	vec3 position;
	vec3 normal;
	vec2 uv;
	float height;
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
	Terrain t_terrain;
};

layout(std140, binding = 4) uniform RenderParametersBuffer
{
	RenderParameters renderParameters;
};

layout(binding = 2) uniform sampler2D t_heightMap;

layout(binding = 8) uniform sampler2D t_windMap;
layout(binding = 9) uniform sampler2D t_resistanceMap;
layout(binding = 10) uniform sampler2D t_moistureMap;
layout(binding = 11) uniform sampler2D t_sedimentMap;


layout(early_fragment_tests) in;
in Fragment fragment;

// Output
layout(location = 0) out vec4 fragmentColor;
layout(location = 1) out vec4 fragmentDepth;

// Functionality
vec3 getAmbientColor()
{
	const vec3 ambientColor = t_environment.ambientIntensity * t_environment.ambientColor;
	return ambientColor;
}

void main()
{
	const vec3 viewVector = t_inverseViewMatrix[3].xyz - fragment.position;
	const float viewDistance = length(viewVector);
	const vec3 viewDirection = viewVector / (viewDistance + EPSILON);
	fragmentDepth = vec4(viewDistance, fragment.position.xz, 1000000.f);

	vec3 normal = normalize(fragment.normal);

	const vec3 ambientColor = getAmbientColor();

	vec3 originalTerrain = texture(t_heightMap, fragment.uv).xzy;
	float terrainSum = 0.f;
	for(int i = 0; i < 3; ++i) {
		terrainSum += originalTerrain[i];
		terrainSum = min(terrainSum, fragment.height);
		originalTerrain[i] = terrainSum;
	}
	const vec3 terrain = vec3(originalTerrain.x, originalTerrain.z - originalTerrain.y, originalTerrain.y - originalTerrain.x);
	const vec4 resistances = texture(t_resistanceMap, fragment.uv);
	const vec3 bedrockColor = mix(renderParameters.bedrockColor.rgb, vec3(0), 0.75 * resistances.z);
	vec3 diffuseColor = mix(renderParameters.soilColor.rgb, bedrockColor, clamp((1.f - terrain.z) / (1.f), 0.f, 1.f));
	diffuseColor = mix(renderParameters.sandColor.rgb, diffuseColor, clamp((1.f - terrain.y) / (1.f), 0.f, 1.f));
	
	fragmentColor.rgb = ambientColor * diffuseColor;

	for (int i = 0; i < t_environment.lightCount; ++i)
	{
		vec3 lightDirection;
		float attenuation;

		if (t_environment.lights[i].type == LIGHT_TYPE_DIRECTIONAL)
		{
			lightDirection = -t_environment.lights[i].direction;
			attenuation = 1.0f;
		}
		else
		{
			const vec3 lightVector = t_environment.lights[i].position - fragment.position;
			const float lightDistance2 = dot(lightVector, lightVector);
			const float lightDistance = sqrt(lightDistance2);

			if (lightDistance >= t_environment.lights[i].range)
			{
				continue;
			}

			lightDirection = lightVector / (lightDistance + EPSILON);
			attenuation = clamp(1.0f / (t_environment.lights[i].attenuation.x +
				                        t_environment.lights[i].attenuation.y * lightDistance +
				                        t_environment.lights[i].attenuation.z * lightDistance2), 0.0f, 1.0f);

			if (t_environment.lights[i].type == LIGHT_TYPE_SPOT)
			{
			    const float cosTheta = dot(lightDirection, -t_environment.lights[i].direction);

				if (cosTheta < t_environment.lights[i].spotOuterCutOff)
				{
				    continue;
				}

				attenuation *= clamp((cosTheta - t_environment.lights[i].spotOuterCutOff) /
					                 (t_environment.lights[i].spotInnerCutOff - t_environment.lights[i].spotOuterCutOff), 0.0f, 1.0f);
			}
		}
		
		const vec3 reflection = reflect(-lightDirection, normal);
		const float cosPhi = max(dot(normal, lightDirection), 0.0f);
		const float cosPsi = max(dot(reflection, viewDirection), 0.0f);
		const vec3 lightColor = attenuation * t_environment.lights[i].intensity * t_environment.lights[i].color;
			
	    const vec3 specularColor = vec3(0);
		const float cosPsiN = pow(cosPsi, 80.0f);
		   
		const vec3 illuminatedColor = lightColor * (cosPhi * diffuseColor + cosPsiN * specularColor);
		
		fragmentColor.rgb += illuminatedColor;
	}

	fragmentColor.rgb = clamp(fragmentColor.rgb, 0.0f, 1.0f);
	fragmentColor.a = 0.f;
}
