#version 460 core

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
	vec3 waterNormal;
	vec2 uv;
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
	Terrain t_terrain;
};

layout(std140, binding = 4) uniform RenderParametersBuffer
{
	RenderParameters renderParameters;
};

layout(binding = 2) uniform sampler2D t_heightMap;

layout(binding = 8) uniform sampler2D t_windMap;
layout(binding = 9) uniform sampler2D t_resistanceMap;

layout(early_fragment_tests) in;
in Fragment fragment;

// Output
layout(location = 0) out vec4 fragmentColor;

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

	vec3 normal = normalize(fragment.normal);
	vec3 waterNormal = normalize(fragment.waterNormal);

	const vec3 ambientColor = getAmbientColor();
	
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


		const vec4 terrain = texture(t_heightMap, fragment.uv);
		const vec4 resistances = texture(t_resistanceMap, fragment.uv);
		const vec3 bedrockColor = mix(renderParameters.bedrockColor.rgb, vec3(0), 0.75 * resistances.z);
		vec3 diffuseColor = mix(renderParameters.soilColor.rgb, bedrockColor, clamp((1.f - terrain.z) / (1.f), 0.f, 1.f));
	    diffuseColor = mix(renderParameters.sandColor.rgb, diffuseColor, clamp((1.f - terrain.y) / (1.f), 0.f, 1.f));
		diffuseColor = mix(diffuseColor, renderParameters.vegetationColor.rgb, max(resistances.y, 0.f));
		//diffuseColor = mix(diffuseColor, renderParameters.objectColor.rgb, max(-resistances.y,0));
			
	    const vec3 specularColor = vec3(0);
		const float cosPsiN = pow(cosPsi, 80.0f);
		   
		fragmentColor.rgb += ambientColor * diffuseColor;
		const vec3 illuminatedColor = lightColor * (cosPhi * diffuseColor + cosPsiN * specularColor);
		//if (resistances.w < 0.0f && renderParameters.erosionColor.a > 0.5f) 
		//{
		//    fragmentColor.rgb += mix(illuminatedColor, illuminatedColor * renderParameters.erosionColor.rgb, 0.5f);
		//}
		//else if (resistances.w > 0.0f && renderParameters.stickyColor.a > 0.5f) 
		//{
		//    fragmentColor.rgb += mix(illuminatedColor, illuminatedColor * renderParameters.stickyColor.rgb, 0.5f * resistances.w);
		//} else 
		{
			fragmentColor.rgb += illuminatedColor;
		}
		if(renderParameters.waterColor.a > 0.5f) {
			const vec3 waterColor = mix(renderParameters.waterColor.rgb, renderParameters.soilColor.rgb, min(10.f * resistances.w, 1.f));
			fragmentColor.rgb = mix(fragmentColor.rgb, illuminatedColor * waterColor, min(0.1f * terrain.w, 1));
			fragmentColor.rgb = mix(fragmentColor.rgb, vec3(0.2) * waterColor, clamp(0.1f * terrain.w - 1.f, 0, 1));

			const float specularFactor = clamp(5.f * terrain.w, 0.f, 1.f);
			const vec3 backgroundColor = 0.95f * vec3(0.7f, 0.9f, 1.0f);
			const float cosTheta = max(dot(waterNormal, viewDirection), 0.0f);
			const float cosTheta5 = pow(1 - cosTheta, 5.f);
			const float rTheta = specularFactor * (0.04 + 0.96 * cosTheta5);

			const vec3 reflection = reflect(-lightDirection, waterNormal);
			const float cosPsi = max(dot(reflection, viewDirection), 0.0f);
			const vec3 specularColor = specularFactor * vec3(1);
			const float cosPsiN = pow(cosPsi, 1600.0f);

			fragmentColor.rgb = mix(fragmentColor.rgb, backgroundColor, rTheta) + cosPsiN * specularColor;
		}
		if (resistances.x > 0.0f && renderParameters.windShadowColor.a > 0.5f)
		{
		    fragmentColor.rgb = mix(fragmentColor.rgb, fragmentColor.rgb * renderParameters.windShadowColor.rgb, max(1.f - terrain.w, 0.f) * 0.5f * resistances.x);
		} 
		//fragmentColor.rgb = vec3(resistances.w);
	}

	fragmentColor.rgb = clamp(fragmentColor.rgb, 0.0f, 1.0f);
}
