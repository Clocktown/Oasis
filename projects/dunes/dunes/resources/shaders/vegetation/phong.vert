#version 460 core
#extension GL_NV_gpu_shader5 : require

// Input
layout(location = 0) in vec4 t_position;
layout(location = 1) in vec4 t_normal;
layout(location = 2) in vec4 t_tangent;
layout(location = 3) in vec2 t_uv;

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

#define PI 3.141592653589793f
// Source: https://gist.github.com/patriciogonzalezvivo/670c22f3966e662d2f83
float rand(vec2 c){
	return fract(sin(dot(c.xy ,vec2(12.9898,78.233))) * 43758.5453);
}

float noise(vec2 p, float freq ){
	float unit = 1.f/freq;
	vec2 ij = floor(p/unit);
	vec2 xy = mod(p,unit)/unit;
	//xy = 3.*xy*xy-2.*xy*xy*xy;
	xy = .5*(1.-cos(PI*xy));
	float a = rand((ij+vec2(0.,0.)));
	float b = rand((ij+vec2(1.,0.)));
	float c = rand((ij+vec2(0.,1.)));
	float d = rand((ij+vec2(1.,1.)));
	float x1 = mix(a, b, xy.x);
	float x2 = mix(c, d, xy.x);
	return mix(x1, x2, xy.y);
}

float pNoise(vec2 p, int res){
	float persistance = .5;
	float n = 0.;
	float normK = 0.;
	float f = 4.;
	float amp = 1.;
	int iCount = 0;
	for (int i = 0; i<50; i++){
		n+=amp*noise(p, f);
		f*=2.;
		normK+=amp;
		amp*=persistance;
		if (iCount == res) break;
		iCount++;
	}
	float nf = n/normK;
	return nf*nf*nf*nf;
}

// Output
struct Fragment
{
	vec3 position;
	vec3 normal;
	vec2 uv;
};

out Fragment fragment;
flat out mat3 tbnMatrix;

// Functionality
void main()
{
    const Vegetation veg = t_vegs[t_vegMap[t_userID.x + gl_InstanceID]];
	const mat3 normalMatrix = mat3(t_modelMatrix);
	const float alpha = PI * pNoise(veg.pos.xy, 1);
	const float ca = cos(alpha);
	const float sa = sin(alpha);
	const mat3 rot = mat3(ca, 0, sa,
						  0,  1, 0,
						  -sa,0, ca);

	fragment.position = (t_modelMatrix * vec4(veg.pos.xzy + veg.radius * rot * t_position.xyz, 1.0f)).xyz;
	fragment.normal = normalize(normalMatrix * t_normal.xyz);
	fragment.uv = t_uv;

	if (t_material.hasNormalMap)
	{
		const vec3 tangent = normalize(normalMatrix * t_tangent.xyz);
		const vec3 bitangent = cross(fragment.normal, tangent);

		tbnMatrix = mat3(tangent, bitangent, fragment.normal);
	}

	gl_Position = t_viewProjectionMatrix * vec4(fragment.position, 1.0f);
}
