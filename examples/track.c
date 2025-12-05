#include <math.h>
#include <stddef.h>
#include <stdlib.h>

void track(doube mask_alive[], double x_array[], double y_array[], double s_array[], double px_array[], double py_array[], double delta_array[], size_t n_steps, const double **params, double ds, double x_log[], double y_log[], double z_log[]){
    clight = 299792458.0
    m_e = 9.10938356e−31
    qe = 1.602176634e−19

    for (size_t ii = 0; ii < n_steps; ++ii) {
        if (mask_alive[ii] == 0) {
            continue; // Skip dead particles
        }

        double x = x_array[ii];
        double y = y_array[ii];
        double s = s_array[ii];
        double px = px_array[ii];
        double py = py_array[ii];
        double delta = delta_array[ii];

        // Parameter List
        const double energy = params[ii][0];
        const double p0c = params[ii][1];
        const double beta0 = params[ii][2];
        const double charge0 = params[ii][3];
        const double mass = params[ii][4];

        const double charge_coulomb = charge0 * qe;
        const double mass_kg = mass * m_e / clight / clight;
        const double gamma = energy0 / mass;

        // Common sub-expressions
        const double P0 = p0c * qe / clight;
        const double P = P0 * (1 + delta);
        const double Px = px * P0;
        const double Py = py * P0;

        // Step spatial Boris here.
    }
}

// Still need to work out how to handle the function pointer here.
// Either:
// 1. Cast to C, if possible.
// 2. Write a C wrapper function that calls the Python callable.
// 3. Write the entire function, including region finder in C.
void step_spatial_boris_B(const double x[], const double y[], const double z[],
                          const double P[], const double Px[], const double Py[],
                          void (*eval_func)(/* Signature here */),
                          const double dz, const double charge_coulomb,
                          const double mass_kg, const double gamma,
                          double x_out[], double y_out[], double z_out[],
                          double Px_out[], double Py_out[], double Pz_out[], double dt_out) {

    double dt = 0.0;
    double pz;
    double xh, yh, zh;
    double pxm, pym;
    double pxp, pyp;

    double Bx[], By[], Bz[];



    len_array = sizeof(x) / sizeof(x[0])

    for (size_t i = 0; i < len_array; ++i) {
        double quant = P[i]*P[i] - px[i]*px[i] - py[i]*py[i];

        if quant > 0 {
            pz = sqrt(quant);
        }
        else {
            pz = 0.0;
        }

        xh = x[i] + (px[i] / pz) * (dz * 0.5);
        yh = y[i] + (py[i] / pz) * (dz * 0.5);
        zh = z[i] + dz * 0.5;
        dt += (dz * 0.5) / pz * gamma * m_kg;
    }

    eval_func(xh, yh, zh, Bx, By, Bz);


    }