#include <cmath>
#include <cstdlib>
#include <SFML/Graphics.hpp>

int N = 128;
int ITER = 1;
int SCALE = 4;

struct Fluid {
	int size;
	float dt;
	float diff;
	float visc;

	float *s;
	float *density;

	float *Vx;
	float *Vy;

	float *Vx0;
	float *Vy0;
};

int IX(int x, int y)
{
	int r = x + (y * N);
	r = r > N * N ? N * N : r;
	r = r < 0 ? 0 : r;
	return r;
}

Fluid *FluidCreate(float diffusion, float viscosity, float dt)
{
	Fluid *plane = (Fluid *)malloc(sizeof(*plane));

	plane->size = N;
	plane->dt = dt;
	plane->diff = diffusion;
	plane->visc = viscosity;

	plane->s = (float *)calloc(N * N, sizeof(float));
	plane->density = (float *)calloc(N * N, sizeof(float));

	plane->Vx = (float *)calloc(N * N, sizeof(float));
	plane->Vy = (float *)calloc(N * N, sizeof(float));

	plane->Vx0 = (float *)calloc(N * N, sizeof(float));
	plane->Vy0 = (float *)calloc(N * N, sizeof(float));

	return plane;
}

void FluidFree(Fluid *cube)
{
	free(cube->s);
	free(cube->density);

	free(cube->Vx);
	free(cube->Vy);

	free(cube->Vx0);
	free(cube->Vy0);

	free(cube);
}

static void set_bnd(int b, float *x)
{

	for (int i = 1; i < N - 1; i++) {
		x[IX(i, 0)] = b == 2 ? -x[IX(i, 1)] : x[IX(i, 1)];
		x[IX(i, N - 1)] = b == 2 ? -x[IX(i, N - 2)] : x[IX(i, N - 2)];
	}

	for (int j = 1; j < N - 1; j++) {
		x[IX(0, j)] = b == 1 ? -x[IX(1, j)] : x[IX(1, j)];
		x[IX(N - 1, j)] = b == 1 ? -x[IX(N - 2, j)] : x[IX(N - 2, j)];
	}

	x[IX(0, 0)] = 0.5f * (x[IX(1, 0)] + x[IX(0, 1)]);
	x[IX(0, N - 1)] = 0.5f * (x[IX(1, N - 1)] + x[IX(0, N - 2)]);
	x[IX(N - 1, 0)] = 0.5f * (x[IX(N - 2, 0)] + x[IX(N - 1, 1)]);
	x[IX(N - 1, N - 1)] = 0.5f * (x[IX(N - 2, N - 1)] + x[IX(N - 1, N - 2)]);
}

static void lin_solve(int b, float *x, float *x0, float a, float c)
{
	float cRecip = 1.0 / c;
	for (int k = 0; k < ITER; k++) {
		for (int j = 1; j < N - 1; j++) {
			for (int i = 1; i < N - 1; i++) {
				x[IX(i, j)] =
					(x0[IX(i, j)]
					 + a * (x[IX(i + 1, j)]
						+ x[IX(i - 1, j)]
						+ x[IX(i, j + 1)]
						+ x[IX(i, j - 1)]
					 )) * cRecip;
			}
		}
		set_bnd(b, x);
	}
}

static void diffuse(int b, float *x, float *x0, float diff, float dt)
{
	float a = dt * diff * (N - 2) * (N - 2);
	lin_solve(b, x, x0, a, 1 + 6 * a);
}

static void advect(int b, float *d, float *d0, float *velocX, float *velocY, float dt)
{
	float i0, i1, j0, j1;

	float dtx = dt * (N - 2);
	float dty = dt * (N - 2);

	float s0, s1, t0, t1;
	float tmp1, tmp2, x, y;

	float Nfloat = N;
	float ifloat, jfloat;
	int i, j;

	for (j = 1, jfloat = 1; j < N - 1; j++, jfloat++) {
		for (i = 1, ifloat = 1; i < N - 1; i++, ifloat++) {
			tmp1 = dtx * velocX[IX(i, j)];
			tmp2 = dty * velocY[IX(i, j)];
			x = ifloat - tmp1;
			y = jfloat - tmp2;

			if (x < 0.5f) x = 0.5f;
			if (x > Nfloat + 0.5f) x = Nfloat + 0.5f;
			i0 = floorf(x);
			i1 = i0 + 1.0f;
			if (y < 0.5f) y = 0.5f;
			if (y > Nfloat + 0.5f) y = Nfloat + 0.5f;
			j0 = floorf(y);
			j1 = j0 + 1.0f;

			s1 = x - i0;
			s0 = 1.0f - s1;
			t1 = y - j0;
			t0 = 1.0f - t1;

			int i0i = i0;
			int i1i = i1;
			int j0i = j0;
			int j1i = j1;

			d[IX(i, j)] =
				s0 * (t0 * d0[IX(i0i, j0i)] + t1 * d0[IX(i0i, j1i)]) +
				s1 * (t0 * d0[IX(i1i, j0i)] + t1 * d0[IX(i1i, j1i)]);
		}
	}
	set_bnd(b, d);
}

static void project(float *velocX, float *velocY, float *p, float *div)
{
	for (int j = 1; j < N - 1; j++) {
		for (int i = 1; i < N - 1; i++) {
			div[IX(i, j)] = -0.5f * (
				velocX[IX(i + 1, j)]
				- velocX[IX(i - 1, j)]
				+ velocY[IX(i, j + 1)]
				- velocY[IX(i, j - 1)]
			) / N;
			p[IX(i, j)] = 0;

		}
	}
	set_bnd(0, div);
	set_bnd(0, p);
	lin_solve(0, p, div, 1, 6);

	for (int j = 1; j < N - 1; j++) {
		for (int i = 1; i < N - 1; i++) {
			velocX[IX(i, j)] -= 0.5f * (p[IX(i + 1, j)]
						    - p[IX(i - 1, j)]) * N;
			velocY[IX(i, j)] -= 0.5f * (p[IX(i, j + 1)]
						    - p[IX(i, j - 1)]) * N;
		}
	}

	set_bnd(1, velocX);
	set_bnd(2, velocY);
}

void FluidStep(Fluid *cube)
{
	float visc = cube->visc;
	float diff = cube->diff;
	float dt = cube->dt;
	float *Vx = cube->Vx;
	float *Vy = cube->Vy;
	float *Vx0 = cube->Vx0;
	float *Vy0 = cube->Vy0;
	float *s = cube->s;
	float *density = cube->density;

	diffuse(1, Vx0, Vx, visc, dt);
	diffuse(2, Vy0, Vy, visc, dt);

	project(Vx0, Vy0, Vx, Vy);

	advect(1, Vx, Vx0, Vx0, Vy0, dt);
	advect(2, Vy, Vy0, Vx0, Vy0, dt);

	project(Vx, Vy, Vx0, Vy0);

	diffuse(0, s, density, diff, dt);
	advect(0, density, s, Vx, Vy, dt);
}

void FluidAddDensity(Fluid *plane, int x, int y, float amount)
{
	float toadd = plane->density[IX(x, y)] + amount;
	plane->density[IX(x, y)] = (toadd >= 255) ? 255.0 : toadd;
}

void FluidAddVelocity(Fluid *plane, int x, int y, float amountX, float amountY)
{
	int index = IX(x, y);

	plane->Vx[index] += amountX;
	plane->Vy[index] += amountY;
}

void FluidFade(Fluid *plane, float amount)
{
	for (int i = 0; i < N * N; ++i) {
		float fadeD = plane->density[i] - amount;
		plane->density[i] = (fadeD <= 0) ? 0.0 : fadeD;
	}
}

auto ColorMap(float value)
{
	float ratio = 2 * value / 255.0f;
	int b = 0 > 255 * (1 - ratio) ? 0 : int(255 * (1 - ratio));
	int r = 0 > 255 * (ratio - 1) ? 0 : int(255 * (ratio - 1));
	int g = 255 - b - r;

	return sf::Color(r, g, b);
}

int main()
{
	sf::RenderWindow window(sf::VideoMode(N * SCALE, N * SCALE), "Fluid");
	window.setFramerateLimit(60);
	auto fluid = FluidCreate(0, 0, 0.2);
	sf::VertexArray grid(sf::Quads);
	int px = 0, py = 0;

	while (window.isOpen()) {
		int x = sf::Mouse::getPosition(window).x / SCALE;
		int y = sf::Mouse::getPosition(window).y / SCALE;

		sf::Event event = {};
		while (window.pollEvent(event)) {
			if (event.type == sf::Event::Closed) { window.close(); }
		}
		grid.clear();
		window.clear(sf::Color::Blue);

		if (sf::Mouse::isButtonPressed(sf::Mouse::Left)) {
			int brush = 3;
			for (int i = -brush; i <= brush; i++) {
				for (int j = -brush; j <= brush; j++) {
					FluidAddDensity(fluid, x + i, y + j, 64);
				}
			}

			FluidAddVelocity(fluid, x, y, x - px, y - py);
		}

		for (int i = 0; i < N; ++i) {
			for (int j = 0; j < N; ++j) {
				float x = i * SCALE;
				float y = j * SCALE;
				float d = fluid->density[IX(i, j)];
				sf::Color c(ColorMap(d));
				if (d > 0) {
					grid.append(sf::Vertex(sf::Vector2f(x, y), c));
					grid.append(sf::Vertex(sf::Vector2f(x + SCALE, y), c));
					grid.append(sf::Vertex(sf::Vector2f(x + SCALE, y + SCALE), c));
					grid.append(sf::Vertex(sf::Vector2f(x, y + SCALE), c));
				}
			}
		}
		FluidStep(fluid);
		FluidFade(fluid, 0.5);
		window.draw(grid);
		window.display();
		px = x;
		py = y;
	}
	FluidFree(fluid);
	return 0;
}
