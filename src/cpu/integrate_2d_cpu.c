/*
 * integrate_2d_cpu.c — CPU reference implementation of 2D numerical integration
 *
 * Same algorithm as the CUDA version but runs sequentially on the CPU.
 * Useful for:
 *   - validating that the GPU result is correct,
 *   - benchmarking speedup  (CPU time / GPU time).
 *
 * The approach:
 *   1. Discretise the bounding box into nx × ny cells of area ΔA = Δx·Δy.
 *   2. For each cell centre (xi, yj), check if it lies inside the domain.
 *   3. If yes, accumulate f(xi, yj) · ΔA.
 */

#include <math.h>
#include "../include/integral_2d.h"

/* The integrand — same function as on the GPU side */
static double func_2d(double x, double y) {
    return x * x + y * y;
}

/* Check whether (x,y) is inside the domain D */
static int point_in_domain(double x, double y, Shape shape,
                           double cx, double cy, double R) {
    if (shape == SHAPE_CIRCLE) {
        double dx = x - cx;
        double dy = y - cy;
        return (dx * dx + dy * dy) <= (R * R);
    } else {
        /* Triangle with vertices (cx, cy+R), (cx-R, cy-R), (cx+R, cy-R) */
        double x0 = cx,     y0 = cy + R;
        double x1 = cx - R, y1 = cy - R;
        double x2 = cx + R, y2 = cy - R;
        double d1 = (x - x1) * (y0 - y1) - (x0 - x1) * (y - y1);
        double d2 = (x - x2) * (y1 - y2) - (x1 - x2) * (y - y2);
        double d3 = (x - x0) * (y2 - y0) - (x2 - x0) * (y - y0);
        int has_neg = (d1 < 0) || (d2 < 0) || (d3 < 0);
        int has_pos = (d1 > 0) || (d2 > 0) || (d3 > 0);
        return !(has_neg && has_pos);
    }
}

/* ── Main CPU integration routine ─────────────────────────────────────────── */
double integrate_2d_cpu(IntegralParams2D p) {
    double dx = (p.x_max - p.x_min) / p.nx;
    double dy = (p.y_max - p.y_min) / p.ny;
    double sum = 0.0;

    /* Simple double loop over all grid cells */
    for (int iy = 0; iy < p.ny; iy++) {
        double y = p.y_min + (iy + 0.5) * dy;        /* cell-centre y */
        for (int ix = 0; ix < p.nx; ix++) {
            double x = p.x_min + (ix + 0.5) * dx;    /* cell-centre x */

            if (point_in_domain(x, y, p.shape, p.center_x, p.center_y, p.radius)) {
                sum += func_2d(x, y) * dx * dy;
            }
        }
    }

    return sum;
}
