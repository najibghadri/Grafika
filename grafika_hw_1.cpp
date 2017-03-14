//=============================================================================================
// Szamitogepes grafika hazi feladat keret. Ervenyes 2017-tol.
// A //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// sorokon beluli reszben celszeru garazdalkodni, mert a tobbit ugyis toroljuk.
// A beadott program csak ebben a fajlban lehet, a fajl 1 byte-os ASCII karaktereket tartalmazhat.
// Tilos:
// - mast "beincludolni", illetve mas konyvtarat hasznalni
// - faljmuveleteket vegezni a printf-et kivéve
// - new operatort hivni a lefoglalt adat korrekt felszabaditasa nelkul
// - felesleges programsorokat a beadott programban hagyni
// - felesleges kommenteket a beadott programba irni a forrasmegjelolest kommentjeit kiveve
// ---------------------------------------------------------------------------------------------
// A feladatot ANSI C++ nyelvu forditoprogrammal ellenorizzuk, a Visual Studio-hoz kepesti elteresekrol
// es a leggyakoribb hibakrol (pl. ideiglenes objektumot nem lehet referencia tipusnak ertekul adni)
// a hazibeado portal ad egy osszefoglalot.
// ---------------------------------------------------------------------------------------------
// A feladatmegoldasokban csak olyan OpenGL/GLUT fuggvenyek hasznalhatok, amelyek az oran a feladatkiadasig elhangzottak 
//
// NYILATKOZAT
// ---------------------------------------------------------------------------------------------
// Nev    : Ghadri Najib
// Neptun : C24J1U
// ---------------------------------------------------------------------------------------------
// ezennel kijelentem, hogy a feladatot magam keszitettem, es ha barmilyen segitseget igenybe vettem vagy
// mas szellemi termeket felhasznaltam, akkor a forrast es az atvett reszt kommentekben egyertelmuen jeloltem.
// A forrasmegjeloles kotelme vonatkozik az eloadas foliakat es a targy oktatoi, illetve a
// grafhazi doktor tanacsait kiveve barmilyen csatornan (szoban, irasban, Interneten, stb.) erkezo minden egyeb
// informaciora (keplet, program, algoritmus, stb.). Kijelentem, hogy a forrasmegjelolessel atvett reszeket is ertem,
// azok helyessegere matematikai bizonyitast tudok adni. Tisztaban vagyok azzal, hogy az atvett reszek nem szamitanak
// a sajat kontribucioba, igy a feladat elfogadasarol a tobbi resz mennyisege es minosege alapjan szuletik dontes.
// Tudomasul veszem, hogy a forrasmegjeloles kotelmenek megsertese eseten a hazifeladatra adhato pontokat
// negativ elojellel szamoljak el es ezzel parhuzamosan eljaras is indul velem szemben.
//=============================================================================================

#define _USE_MATH_DEFINES
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include <vector>

#if defined(__APPLE__)
#include <GLUT/GLUT.h>
#include <OpenGL/gl3.h>
#include <OpenGL/glu.h>
#else
#if defined(WIN32) || defined(_WIN32) || defined(__WIN32__)
#include <windows.h>
#endif
#include <GL/glew.h>		// must be downloaded 
#include <GL/freeglut.h>	// must be downloaded unless you have an Apple
#endif


const unsigned int windowWidth = 600, windowHeight = 600;

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// You are supposed to modify the code from here...

// OpenGL major and minor versions
int majorVersion = 3, minorVersion = 3;

void getErrorInfo(unsigned int handle) {
	int logLen;
	glGetShaderiv(handle, GL_INFO_LOG_LENGTH, &logLen);
	if (logLen > 0) {
		char * log = new char[logLen];
		int written;
		glGetShaderInfoLog(handle, logLen, &written, log);
		printf("Shader log:\n%s", log);
		delete log;
	}
}

// check if shader could be compiled
void checkShader(unsigned int shader, char * message) {
	int OK;
	glGetShaderiv(shader, GL_COMPILE_STATUS, &OK);
	if (!OK) {
		printf("%s!\n", message);
		getErrorInfo(shader);
	}
}

// check if shader could be linked
void checkLinking(unsigned int program) {
	int OK;
	glGetProgramiv(program, GL_LINK_STATUS, &OK);
	if (!OK) {
		printf("Failed to link shader program!\n");
		getErrorInfo(program);
	}
}

// vertex shader in GLSL
const char * vertexSource = R"(
	#version 330
    precision highp float;

	uniform mat4 MVP;			// Model-View-Projection matrix in row-major format

	layout(location = 0) in vec2 vertexPosition;	// Attrib Array 0
	layout(location = 1) in vec3 vertexColor;	    // Attrib Array 1
	out vec3 color;									// output attribute

	void main() {
		color = vertexColor;														// copy color from input to output
		gl_Position = vec4(vertexPosition.x, vertexPosition.y, 0, 1) * MVP; 		// transform to clipping space
	}
)";

// fragment shader in GLSL
const char * fragmentSource = R"(
	#version 330
    precision highp float;

	in vec3 color;				// variable input: interpolated color of vertex shader
	out vec4 fragmentColor;		// output that goes to the raster memory as told by glBindFragDataLocation

	void main() {
		fragmentColor = vec4(color, 1); // extend RGB to RGBA
	}
)";

// row-major matrix 4x4
struct mat4 {
	float m[4][4];
public:
	mat4() {}
	mat4(float m00, float m01, float m02, float m03,
		float m10, float m11, float m12, float m13,
		float m20, float m21, float m22, float m23,
		float m30, float m31, float m32, float m33) {
		m[0][0] = m00; m[0][1] = m01; m[0][2] = m02; m[0][3] = m03;
		m[1][0] = m10; m[1][1] = m11; m[1][2] = m12; m[1][3] = m13;
		m[2][0] = m20; m[2][1] = m21; m[2][2] = m22; m[2][3] = m23;
		m[3][0] = m30; m[3][1] = m31; m[3][2] = m32; m[3][3] = m33;
	}

	mat4 operator*(const mat4& right) {
		mat4 result;
		for (int i = 0; i < 4; i++) {
			for (int j = 0; j < 4; j++) {
				result.m[i][j] = 0;
				for (int k = 0; k < 4; k++) result.m[i][j] += m[i][k] * right.m[k][j];
			}
		}
		return result;
	}

	operator float*() { return &m[0][0]; }


};


// 3D point in homogeneous coordinates
struct vec4 {
	float v[4];

	vec4(float x = 0, float y = 0, float z = 0, float w = 1) {
		v[0] = x; v[1] = y; v[2] = z; v[3] = w;
	}

	vec4& operator+=(const vec4& right) {
		*this = vec4(v[0] += right.v[0], v[1] += right.v[1], v[2] += right.v[2], v[3] += right.v[3]);
		return *this;
	}
	vec4 operator*(const mat4& mat) {
		vec4 result;
		for (int j = 0; j < 4; j++) {
			result.v[j] = 0;
			for (int i = 0; i < 4; i++) result.v[j] += v[i] * mat.m[i][j];
		}
		return result;
	}
	vec4 operator*(float a) const {
		vec4 res(v[0] * a, v[1] * a, v[2] * a);
		return res;
	}
	vec4 operator+(const vec4& right) const {
		return vec4(v[0] + right.v[0], v[1] + right.v[1], v[2] + right.v[2], v[3] + right.v[3]);
	}
	vec4 operator-(const vec4& right) const {
		return vec4(v[0] - right.v[0], v[1] - right.v[1], v[2] - right.v[2], v[3] - right.v[3]);
	}
	vec4 operator*(const vec4& right) const {
		return vec4(v[0] * right.v[0], v[1] * right.v[1], v[2] * right.v[2], v[3] * right.v[3]);
	}
	float dot(const vec4& right) const {
		return (v[0] * right.v[0] + v[1] * right.v[1] + v[2] * right.v[2]);
	}
	float length() const {
		return sqrtf(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
	}
};


float WSize = 100; //100 which is 1 km / 10m (thus 50m is 5)

// 2D camera
struct Camera {
	float wCx, wCy;	// center in world coordinates
	float wWx, wWy;	// width and height in world coordinates
public:
	Camera() {
		Animate(0);
	}

	mat4 V() { // view matrix: translates the center to the origin
		return mat4(1, 0, 0, 0,
			0, 1, 0, 0,
			0, 0, 1, 0,
			-wCx, -wCy, 0, 1);
	}

	mat4 P() { // projection matrix: scales it to be a square of edge length 2
		return mat4(2 / wWx, 0, 0, 0,
			0, 2 / wWy, 0, 0,
			0, 0, 1, 0,
			0, 0, 0, 1);
	}

	mat4 Vinv() { // inverse view matrix
		return mat4(1, 0, 0, 0,
			0, 1, 0, 0,
			0, 0, 1, 0,
			wCx, wCy, 0, 1);
	}

	mat4 Pinv() { // inverse projection matrix
		return mat4(wWx / 2, 0, 0, 0,
			0, wWy / 2, 0, 0,
			0, 0, 1, 0,
			0, 0, 0, 1);
	}

	void Animate(float t) {
		wCx = WSize / 2;
		wCy = WSize / 2;
		wWx = WSize;
		wWy = WSize;
	}
};

// 2D camera
Camera camera;

// handle of the shader program
unsigned int shaderProgram;

////////////////////////////////////////////////
// Models
////////////////////////////////////////////////
#define DEBUG

class BezierSurface {
	GLuint vao, vbo;		// vertex array object, vertex buffer object
	float wVertexData[5000];	// data of coordinates and colors
	int nVertices;

	//Control points gird size
	int cpsSize = 5;
	vec4 const cps[5][5] = {
		vec4(0,	100,	0),		vec4(25,	100,	0),		vec4(50, 100,	0),		vec4(75,	100,	0),		 vec4(100,	100,0),
		vec4(0,	75,		0),		vec4(25,	75,		60),	vec4(50, 75,	70),	vec4(75,	75,		40),	 vec4(100,	75,	10),
		vec4(0,	50,		30),	vec4(25,	50,		200),	vec4(50, 50,	200),	vec4(75,	50,		40),	 vec4(100,	50,	60),
		vec4(0,	25,		30),	vec4(25,	25,		70),	vec4(50, 25,	40),	vec4(75,	25,		50),	 vec4(100,	25,	60),
		vec4(0,	0,		60),	vec4(25,	0,		40),	vec4(50, 0,		0),		vec4(75,	0,		30),	 vec4(100,	0,	100),
	};	// vertex data on the CPU

	vec4 zColorInterp(const vec4& vec) {
		float z = vec.v[2];

		float blue = (z / 100); //
		float green = 1 - blue; //

		return vec4(0, green, blue, 0);
	}

	float B(int i, float t) {
		int n = cpsSize - 1; // n deg polynomial = n+1 pts!
		float choose = 1;
		for (int j = 1; j <= i; j++) choose *= (float)(n - j + 1) / j;
		return choose * pow(t, i) * pow(1 - t, n - i);
	}

	vec4 BS(float u, float v) {
		u = u / 100;
		v = v / 100;
		vec4 rr(0, 0, 0);
		for (int n = 0; n < cpsSize; n++)
			for (int m = 0; m < cpsSize; m++)
				rr += cps[n][m] * B(n, u) * B(m, v);
		return rr;
	}

	float dB(int i, float t) {
		int n = cpsSize - 2; // n deg polynomial = n+1 pts, -1 for derivative equation
		float choose = 1;
		for (int j = 1; j <= i; j++) choose *= (float)(n - j + 1) / j;
		return choose * pow(t, i) * pow(1 - t, n - i);
	}

	vec4 dBS(float u, float v) {
		u = u / 100;
		v = v / 100;
		vec4 rr(0, 0, 0);
		for (int n = 0; n < cpsSize - 1; n++) 
			for (int m = 0; m < cpsSize - 1; m++)
				rr += (cps[n+1][m + 1] - cps[n][m]) * dB(n, u) * dB(m, v);

		return rr;
	}

public:
	vec4 wBS(float x, float y) {
		return BS(100-y,x);
	}
	vec4 wdBS(float x, float y) {
		return dBS(100 - y, x);
	}

	void Create() {
		glGenVertexArrays(1, &vao);
		glBindVertexArray(vao);

		glGenBuffers(1, &vbo); // Generate 1 vertex buffer object
		glBindBuffer(GL_ARRAY_BUFFER, vbo);

		// Enable the vertex attribute arrays
		glEnableVertexAttribArray(0);  // attribute array 0
		glEnableVertexAttribArray(1);  // attribute array 1

		// Map attribute array 0 to the vertex data of the interleaved vbo
		// attribute array, components/attribute, component type, normalize?, stride, offset
		glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 5 * sizeof(float), reinterpret_cast<void*>(0));

		// Map attribute array 1 to the color data of the interleaved vbo
		glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 5 * sizeof(float), reinterpret_cast<void*>(2 * sizeof(float)));

		nVertices = 0;
		vec4 wNextVec;
		vec4 nextCol;
		
		int u, v;
		//World size is 100 and model resolution is 5:
		for (u = 0; u <= 100 - 5; u += 5) {
			for (v = 0; v <= 100; v += 5) {
				wNextVec = BS(u, v);
				nextCol = zColorInterp(wNextVec);
				// triangle left bot (and right bot)
				wVertexData[5 * nVertices] = wNextVec.v[0];
				wVertexData[5 * nVertices + 1] = wNextVec.v[1];
				wVertexData[5 * nVertices + 2] = nextCol.v[0]; // red
				wVertexData[5 * nVertices + 3] = nextCol.v[1]; // green
				wVertexData[5 * nVertices + 4] = nextCol.v[2]; // blue
				nVertices++;

				wNextVec = BS(u + 5, v);
				nextCol = zColorInterp(wNextVec);
				// triangle left top (and right top)
				wVertexData[5 * nVertices] = wNextVec.v[0];
				wVertexData[5 * nVertices + 1] = wNextVec.v[1];
				wVertexData[5 * nVertices + 2] = nextCol.v[0]; // red
				wVertexData[5 * nVertices + 3] = nextCol.v[1]; // green
				wVertexData[5 * nVertices + 4] = nextCol.v[2]; // blue
				nVertices++;
			}
			wNextVec = BS(u + 5, v - 5);
			nextCol = zColorInterp(wNextVec);
			// first terminal
			wVertexData[5 * nVertices] = wNextVec.v[0];
			wVertexData[5 * nVertices + 1] = wNextVec.v[1];
			wVertexData[5 * nVertices + 2] = nextCol.v[0]; // red
			wVertexData[5 * nVertices + 3] = nextCol.v[1]; // green
			wVertexData[5 * nVertices + 4] = nextCol.v[2]; // blue
			nVertices++;

			wNextVec = BS(u + 5 , 0);
			nextCol = zColorInterp(wNextVec);
			// first starter)
			wVertexData[5 * nVertices] = wNextVec.v[0];
			wVertexData[5 * nVertices + 1] = wNextVec.v[1];
			wVertexData[5 * nVertices + 2] = nextCol.v[0]; // red
			wVertexData[5 * nVertices + 3] = nextCol.v[1]; // green
			wVertexData[5 * nVertices + 4] = nextCol.v[2]; // blue
			nVertices++;
		}

		// copy data to the GPU
		glBufferData(GL_ARRAY_BUFFER, nVertices * 5 * sizeof(float), wVertexData, GL_STATIC_DRAW);
	}

	void Draw() {

		mat4 VPTransform = camera.V() * camera.P();

		// set GPU uniform matrix variable MVP with the content of CPU variable MVPTransform
		int location = glGetUniformLocation(shaderProgram, "MVP");
		if (location >= 0) glUniformMatrix4fv(location, 1, GL_TRUE, VPTransform); // set uniform variable MVP to the MVPTransform
		else printf("uniform MVP cannot be set\n");

		glBindVertexArray(vao);	// make the vao and its vbos active playing the role of the data source
		glDrawArrays(GL_TRIANGLE_STRIP, 0, nVertices);
	}
};

BezierSurface bezierSurface;

class LagrangeRoute {
	GLuint vao, vbo;			// vertex array object, vertex buffer object
	float  vertexData[150000];	// interleaved data of coordinates and colors
	int    nVertices;			// number of vertices

	bool rLock;					//Lock when cyclist goes through





///////////////////////////////////////
// Lagrange curves
//////////////////
	std::vector<vec4>  cps;		// control points
	std::vector<float> its; 	// incremental (knot) values

//INCREMENTAL LAGRANGE INTERPOLATION


//TIME BASED LAGRANGE INTERPOLATION
public:
	std::vector<float> ts; 	// time (knot) values

	//CPS Lagrange interpolation by time
	vec4 time2IncrementL2CPL(float t) {
		vec4 rr(0, 0, 0);
		for (int i = 0; i < cps.size(); i++) rr += cps[i] * l_base_Time2IncrementL(i, t);
		return rr;
	}
private:
	float l_base_Time2IncrementL(int i, float time) {
		float Li = 1.0f;
		for (int j = 0; j < cps.size(); j++)
			if (j != i) Li *= (time2IncrementL(time) - its[j]) / (its[i] - its[j]);
		return Li;
	}

public:
	//Lagrange curve first derivative
	vec4 time2IncrementL2CPL_D(float time) {
		vec4 rr(0, 0, 0);
		for (int i = 0; i < cps.size(); i++) rr += cps[i] * ld_base_Time2IncrementL(i, time);
		return rr;
	}

private:
	float ld_base_Time2IncrementL(int i, float time) {
		float res = 0;
		for (int j = 0; j < cps.size(); j++)
			if (j != i) res += 1 / (time2IncrementL(time) - its[j]);
		res *= l_base_Time2IncrementL(i, time);
		return res;
	}

	//Increment lagrange interpolation by time
	float time2IncrementL(float time) {
		float rr = 0;
		for (int i = 0; i < its.size(); i++) rr += its[i] * l_base_Time(i, time);
		return rr;
	}

	float l_base_Time(int i, float time) {
		float Li = 1.0f;
		for (int j = 0; j < ts.size(); j++)
			if (j != i) Li *= (time - ts[j]) / (ts[i] - ts[j]);
		return Li;
	}
//////////////////
// Lagrange curves end
///////////////////////////////////////
public:

	void Create() {
		rLock = false;
		glGenVertexArrays(1, &vao);
		glBindVertexArray(vao);

		glGenBuffers(1, &vbo); // Generate 1 vertex buffer object
		glBindBuffer(GL_ARRAY_BUFFER, vbo);
		// Enable the vertex attribute arrays
		glEnableVertexAttribArray(0);  // attribute array 0
		glEnableVertexAttribArray(1);  // attribute array 1
									   // Map attribute array 0 to the vertex data of the interleaved vbo
		glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 5 * sizeof(float), reinterpret_cast<void*>(0)); // attribute array, components/attribute, component type, normalize?, stride, offset
																										// Map attribute array 1 to the color data of the interleaved vbo
		glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 5 * sizeof(float), reinterpret_cast<void*>(2 * sizeof(float)));
	}
	
	void lock() {
		rLock = true;
	}
	void unlock() {
		rLock = false;
	}
	
	void AddPoint(float cX, float cY, float sec) {
		glBindBuffer(GL_ARRAY_BUFFER, vbo);
		if (cps.size() >= 15) return;
		if (rLock == true) return;

#ifdef DEBUG
		system("cls");
#endif

		//Create the new control point from the click params
		vec4 wVec = vec4(cX, cY, 0, 1) * camera.Pinv() * camera.Vinv();

		//its.push_back(cps.size());		//push incremental knot value back
		cps.push_back(wVec);			//push control point back
		ts.push_back(sec);				//push time knot value back

		nVertices = 0;
		vec4 wItVec;
		float ival = (ts.back() - ts.front()) / 1000;

		for (float i = ts.front(); i < ts.back(); i += ival) {
			wItVec = time2IncrementL2CPL(i);

			// fill interleaved data
			vertexData[5 * nVertices] = wItVec.v[0];
			vertexData[5 * nVertices + 1] = wItVec.v[1];

#ifdef DEBUG 
			if (i <= ts.front()+0.05 || i+0.05 >= ts.back()) {
				vertexData[5 * nVertices + 2] = 1; // red
				vertexData[5 * nVertices + 3] = 0; // green
				vertexData[5 * nVertices + 4] = 0; // blue
			}
			else {
				vertexData[5 * nVertices + 2] = 1; // red
				vertexData[5 * nVertices + 3] = 1; // green
				vertexData[5 * nVertices + 4] = 1; // blue
			}
#else
			vertexData[5 * nVertices + 2] = 1; // red
			vertexData[5 * nVertices + 3] = 1; // green
			vertexData[5 * nVertices + 4] = 1; // blue
#endif
			nVertices++;
		}

#ifdef DEBUG
		printf("cX:\t%4.3f\tcY:\t%4.3f\n", cX, cY);
		printf("wVec:\t\t%4.1f\t%4.1f\t%4.1f\n", wVec.v[0], wVec.v[1], wVec.v[2]);
		printf("height:\t%4.1f\n", bezierSurface.wBS(wVec.v[0], wVec.v[1]).v[2]);
		printf("nVertices:\t%d\n", nVertices);
		printf("-------------------\n");
		printf("cps size:\t%d\n", cps.size());
		printf("cps[]:\n");
		for (int i = 0; i < cps.size(); i++)
			printf("%4.1f, %4.1f, %4.1f\t", cps[i].v[0], cps[i].v[1], cps[i].v[2]);
		printf("\n");
		printf("-------------------\n");
		printf("ts size:\t%d\n", ts.size());
		printf("ts[]:\n");
		for (int i = 0; i < ts.size(); i++)
			printf("%4.2f\t", ts[i]);
		printf("\n");
		//printf("-------------------\n");
		//printf("its size:\t%d\n", its.size());
		//printf("its[]:\n");
		//for (int i = 0; i < its.size(); i++)
		//	printf("%4.2f\t", its[i]);
		//printf("\n");

#endif
		// copy data to the GPU
		glBufferData(GL_ARRAY_BUFFER, nVertices * 5 * sizeof(float), vertexData, GL_DYNAMIC_DRAW);
	}

	void Draw() {
		if (nVertices > 0)	{
			mat4 VPTransform = camera.V() * camera.P();

			int location = glGetUniformLocation(shaderProgram, "MVP");
			if (location >= 0) glUniformMatrix4fv(location, 1, GL_TRUE, VPTransform);
			else printf("uniform MVP cannot be set\n");

			glBindVertexArray(vao);
			glDrawArrays(GL_LINE_STRIP, 0, nVertices);
		}
	}
};

LagrangeRoute route;

class Cyclist {
	unsigned int vao;	// vertex array object id
	float sx, sy;		// scaling
	float wTx, wTy;		// translation
	float r11, r12, r21, r22;		// translation

	float time;
	float tDiff;
	bool active;
public:
	Cyclist() {
		active = false;
		sx = 1; sy = 1;
		wTx = 0; wTy = 0;
	}

	void Start(float startTime) {
		route.lock();
		active = true;
		time = route.ts[0];
		tDiff = startTime - time;
		Animate(tDiff);
	}

	void Create() {
		glGenVertexArrays(1, &vao);	// create 1 vertex array object
		glBindVertexArray(vao);		// make it active
		unsigned int vbo[2];		// vertex buffer objects
		glGenBuffers(2, &vbo[0]);	// Generate 2 vertex buffer objects
									// Done with the makin part, baby

									// vertex coordinates: vbo[0] -> Attrib Array 0 -> vertexPosition of the vertex shader
		glBindBuffer(GL_ARRAY_BUFFER, vbo[0]); // make it active, it is an array
		static float vertexCoords[] = { 0,5,		-2, -5,		0, 0,		0, 5,		2, -5,		0, 0 };	// vertex data on the CPU
		glBufferData(GL_ARRAY_BUFFER,   // copy to the GPU
			sizeof(vertexCoords),		// number of the vbo in bytes
			vertexCoords,				// address of the data array on the CPU
			GL_STATIC_DRAW);			// copy to that part of the memory which is not modified 
										// Map Attribute Array 0 to the current bound vertex buffer (vbo[0])
		glEnableVertexAttribArray(0);
		// Data organization of Attribute Array 0 
		glVertexAttribPointer(0,	// Attribute Array 0
			2, GL_FLOAT,			// components/attribute, component type
			GL_FALSE,				// not in fixed point format, do not normalized
			0, NULL);				// stride and offset: it is tightly packed

									// vertex colors: vbo[1] -> Attrib Array 1 -> vertexColor of the vertex shader
		glBindBuffer(GL_ARRAY_BUFFER, vbo[1]);	// make it active, it is an array
		static float vertexColors[] = { 1, 0, 0,	 0, 0, 0,	 1, 0, 0,	 1, 0, 0,	 0, 0, 0,		1, 0, 0};						// vertex data on the CPU
		glBufferData(GL_ARRAY_BUFFER, sizeof(vertexColors), vertexColors, GL_STATIC_DRAW);	// copy to the GPU
																							// Map Attribute Array 1 to the current bound vertex buffer (vbo[1])
		glEnableVertexAttribArray(1);			// Vertex position
												// Data organization of Attribute Array 1
		glVertexAttribPointer(1,	// Attribute Array 1,
			3, GL_FLOAT,			// components/attribute, component type,
			GL_FALSE,				// normalize?, 
			0, NULL);				// tightly packed
	}

	void Animate(float curT) {
		if (active) {
			time = curT - tDiff;
			vec4 pos = route.time2IncrementL2CPL(time);
			vec4 sPos = bezierSurface.wBS(pos.v[0], pos.v[1]);
			vec4 dirV = route.time2IncrementL2CPL_D(time);
			vec4 grad = bezierSurface.wdBS(pos.v[0], pos.v[1]);
			dirV.v[2] = grad.v[2];

			wTx = pos.v[0];
			wTy = pos.v[1];

			float theta = acos(dirV.dot(vec4(0, 1, 0)) / (dirV.length()));

			r11 = cos(theta);
			r12 = -sin(theta);
			r21 = sin(theta);
			r22 = cos(theta);

#ifdef DEBUG
			system("cls");
			printf("sTime:\t%4.3f\n", tDiff);
			printf("curT:\t%4.3f\n", curT);
			printf("time:\t%4.3f\n", time);
			printf("xypos:\t\t%4.1f\t%4.1f\t%4.1f\n", pos.v[0], pos.v[1], pos.v[2]);
			printf("3DPos:\t\t%4.1f\t%4.1f\t%4.1f\n", sPos.v[0], sPos.v[1], sPos.v[2]);
			printf("dirVec:\t\t%4.1f\t%4.1f\t%4.1f\n", dirV.v[0], dirV.v[1], dirV.v[2]);
			printf("grad:\t\t%4.1f\t%4.1f\t%4.1f\n", grad.v[0], grad.v[1], grad.v[2]);
			vec4 temp = vec4(dirV.v[0], dirV.v[1], 0);
			printf("TILT:\t%4.2f", acos(dirV.dot(temp) / (temp.length()*dirV.length())));
#endif
			if (time >= route.ts.back()) {
				active = false;
				route.unlock();
			}
		}
	}

	void Draw() {
		if (!active) return;

		mat4 Mrotate(r11, r12, 0, 0,   //rotation
					r21, r22, 0, 0,
					0, 0, 0, 0,
					0, 0, 0, 1); // model matrix
		
		mat4 Mscale(sx, 0, 0, 0,
					0, sy, 0, 0,
					0, 0, 0, 0,
					0, 0, 0, 1); // model matrix

		mat4 Mtranslate(1, 0, 0, 0,
						0, 1, 0, 0,
						0, 0, 0, 0,
						wTx, wTy, 0, 1); // model matrix

		mat4 MVPTransform = Mrotate * Mscale * Mtranslate * camera.V() * camera.P();

		// set GPU uniform matrix variable MVP with the content of CPU variable MVPTransform
		int location = glGetUniformLocation(shaderProgram, "MVP");
		if (location >= 0) glUniformMatrix4fv(location, 1, GL_TRUE, MVPTransform); // set uniform variable MVP to the MVPTransform
		else printf("uniform MVP cannot be set\n");

		glBindVertexArray(vao);	// make the vao and its vbos active playing the role of the data source
		glDrawArrays(GL_TRIANGLES, 0, 6);	// draw a single triangle with vertices defined in vao
	}
};

Cyclist cyclist;

////////////////////////////////////////////////
// Initialization and events
////////////////////////////////////////////////

// Initialization, create an OpenGL context
void onInitialization() {
	glViewport(0, 0, windowWidth, windowHeight);

	// Create objects by setting up their vertex data on the GPU
	route.Create();
	bezierSurface.Create();
	cyclist.Create();

	// Create vertex shader from string
	unsigned int vertexShader = glCreateShader(GL_VERTEX_SHADER);
	if (!vertexShader) {
		printf("Error in vertex shader creation\n");
		exit(1);
	}
	glShaderSource(vertexShader, 1, &vertexSource, NULL);
	glCompileShader(vertexShader);
	checkShader(vertexShader, "Vertex shader error");

	// Create fragment shader from string
	unsigned int fragmentShader = glCreateShader(GL_FRAGMENT_SHADER);
	if (!fragmentShader) {
		printf("Error in fragment shader creation\n");
		exit(1);
	}
	glShaderSource(fragmentShader, 1, &fragmentSource, NULL);
	glCompileShader(fragmentShader);
	checkShader(fragmentShader, "Fragment shader error");

	// Attach shaders to a single program
	shaderProgram = glCreateProgram();
	if (!shaderProgram) {
		printf("Error in shader program creation\n");
		exit(1);
	}
	glAttachShader(shaderProgram, vertexShader);
	glAttachShader(shaderProgram, fragmentShader);

	// Connect the fragmentColor to the frame buffer memory
	glBindFragDataLocation(shaderProgram, 0, "fragmentColor");	// fragmentColor goes to the frame buffer memory

																// program packaging
	glLinkProgram(shaderProgram);
	checkLinking(shaderProgram);
	// make this program run
	glUseProgram(shaderProgram);
}

void onExit() {
	glDeleteProgram(shaderProgram);
	printf("exit");
}

// Window has become invalid: Redraw
void onDisplay() {
	glClearColor(0, 0, 0, 0);							// background color 
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT); // clear the screen
	
	bezierSurface.Draw();
	route.Draw();
	cyclist.Draw();

	glutSwapBuffers();									// exchange the two buffers
}

// Key of ASCII code pressed
void onKeyboard(unsigned char key, int pX, int pY) {
	long time = glutGet(GLUT_ELAPSED_TIME); // elapsed time since the start of the program
	float sec = time / 1000.0f;				// convert msec to sec
	if (key == ' ') cyclist.Start(sec);         // if d, invalidate display, i.e. redraw
}

// Key of ASCII code released
void onKeyboardUp(unsigned char key, int pX, int pY) {

}

// Mouse click event
void onMouse(int button, int state, int pX, int pY) {
	if (button == GLUT_LEFT_BUTTON && state == GLUT_DOWN) {  // GLUT_LEFT_BUTTON / GLUT_RIGHT_BUTTON and GLUT_DOWN / GLUT_UP
		float cX = 2.0f * pX / windowWidth - 1;	// flip y axis
		float cY = 1.0f - 2.0f * pY / windowHeight;

		long time = glutGet(GLUT_ELAPSED_TIME); // elapsed time since the start of the program
		float sec = time / 1000.0f;				// convert msec to sec

		route.AddPoint(cX, cY, sec);
		glutPostRedisplay();     // redraw
	}
}

// Move mouse with key pressed
void onMouseMotion(int pX, int pY) {
}

// Idle event indicating that some time elapsed: do animation here
void onIdle() {
	long time = glutGet(GLUT_ELAPSED_TIME); // elapsed time since the start of the program
	float sec = time / 1000.0f;				// convert msec to sec
	cyclist.Animate(sec);					// animate the camera
	glutPostRedisplay();					// redraw the scene
}

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// Do not touch the code below this line

int main(int argc, char * argv[]) {
	glutInit(&argc, argv);
#if !defined(__APPLE__)
	glutInitContextVersion(majorVersion, minorVersion);
#endif
	glutInitWindowSize(windowWidth, windowHeight);				// Application window is initially of resolution 600x600
	glutInitWindowPosition(100, 100);							// Relative location of the application window
#if defined(__APPLE__)
	glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE | GLUT_DEPTH | GLUT_3_3_CORE_PROFILE);  // 8 bit R,G,B,A + double buffer + depth buffer
#else
	glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE | GLUT_DEPTH);
#endif
	glutCreateWindow(argv[0]);

#if !defined(__APPLE__)
	glewExperimental = true;	// magic
	glewInit();
#endif

	printf("GL Vendor    : %s\n", glGetString(GL_VENDOR));
	printf("GL Renderer  : %s\n", glGetString(GL_RENDERER));
	printf("GL Version (string)  : %s\n", glGetString(GL_VERSION));
	glGetIntegerv(GL_MAJOR_VERSION, &majorVersion);
	glGetIntegerv(GL_MINOR_VERSION, &minorVersion);
	printf("GL Version (integer) : %d.%d\n", majorVersion, minorVersion);
	printf("GLSL Version : %s\n", glGetString(GL_SHADING_LANGUAGE_VERSION));

	onInitialization();

	glutDisplayFunc(onDisplay);                // Register event handlers
	glutMouseFunc(onMouse);
	glutIdleFunc(onIdle);
	glutKeyboardFunc(onKeyboard);
	glutKeyboardUpFunc(onKeyboardUp);
	//glutMotionFunc(onMouseMotion);

	glutMainLoop();
	onExit();
	return 1;
}