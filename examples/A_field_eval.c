#include <math.h>
#include <stddef.h>

// Auto-generated symbolic field expressions for A
void evaluate_A(const double x_array[], const double y_array[], const double s_array[], size_t n, const double **params, double Ax_out[], double Ay_out[], double As_out[]){
	for (size_t ii = 0; ii < n; ++ii) {
		const double x = x_array[ii];
		const double y = y_array[ii];
		const double s = s_array[ii];

		// Parameter List
		const double bs_0 = params[ii][0];
		const double bs_1 = params[ii][1];
		const double bs_2 = params[ii][2];
		const double bs_3 = params[ii][3];
		const double bs_4 = params[ii][4];
		const double a_1_0 = params[ii][5];
		const double a_1_1 = params[ii][6];
		const double a_1_2 = params[ii][7];
		const double a_1_3 = params[ii][8];
		const double a_1_4 = params[ii][9];
		const double a_2_0 = params[ii][10];
		const double a_2_1 = params[ii][11];
		const double a_2_2 = params[ii][12];
		const double a_2_3 = params[ii][13];
		const double a_2_4 = params[ii][14];
		const double a_3_0 = params[ii][15];
		const double a_3_1 = params[ii][16];
		const double a_3_2 = params[ii][17];
		const double a_3_3 = params[ii][18];
		const double a_3_4 = params[ii][19];
		const double b_1_0 = params[ii][20];
		const double b_1_1 = params[ii][21];
		const double b_1_2 = params[ii][22];
		const double b_1_3 = params[ii][23];
		const double b_1_4 = params[ii][24];
		const double b_2_0 = params[ii][25];
		const double b_2_1 = params[ii][26];
		const double b_2_2 = params[ii][27];
		const double b_2_3 = params[ii][28];
		const double b_2_4 = params[ii][29];
		const double b_3_0 = params[ii][30];
		const double b_3_1 = params[ii][31];
		const double b_3_2 = params[ii][32];
		const double b_3_3 = params[ii][33];
		const double b_3_4 = params[ii][34];

		// Common sub-expressions
		const double x0 = y*y;
		const double x1 = (1.0/2.0)*b_2_1;
		const double x2 = s*x;
		const double x3 = s*s;
		const double x4 = (3.0/2.0)*x3;
		const double x5 = s*s*s;
		const double x6 = 2*x5;
		const double x7 = x*x;
		const double x8 = (1.0/2.0)*x7;
		const double x9 = s*x8;
		const double x10 = s*s*s*s;
		const double x11 = x*x*x;
		const double x12 = x*x3;
		const double x13 = x*x5;
		const double x14 = (1.0/2.0)*x3;
		const double x15 = (1.0/2.0)*b_2_0 + b_2_2*x14 + (1.0/2.0)*b_2_3*x5 + (1.0/2.0)*b_2_4*x10 + s*x1;
		const double x16 = x*x10;

		// Reduced expressions
		Ax_out[ii] = -x0*((1.0/2.0)*b_1_1 + b_1_2*s + b_1_3*x4 + b_1_4*x6 + b_2_2*x2 + b_2_3*x*x4 + b_2_4*x*x6 + (1.0/4.0)*b_3_1*x7 + b_3_2*x9 + (3.0/4.0)*b_3_3*x3*x7 + b_3_4*x5*x7 + x*x1) - y*(a_1_1*x + 2*a_1_2*x2 + 3*a_1_3*x12 + 4*a_1_4*x13 + a_2_1*x8 + a_2_2*s*x7 + a_2_3*x4*x7 + a_2_4*x6*x7 + (1.0/6.0)*a_3_1*x11 + (1.0/3.0)*a_3_2*s*x11 + a_3_3*x11*x14 + (2.0/3.0)*a_3_4*x11*x5 + bs_0 + bs_1*s + bs_2*x3 + bs_3*x5 + bs_4*x10);
		Ay_out[ii] = 0;
		As_out[ii] = -x*(b_1_0 + b_1_1*s + b_1_2*x3 + b_1_3*x5 + b_1_4*x10) + x0*((1.0/2.0)*b_3_0*x + (1.0/2.0)*b_3_1*x2 + (1.0/2.0)*b_3_2*x12 + (1.0/2.0)*b_3_3*x13 + (1.0/2.0)*b_3_4*x16 + x15) - x11*((1.0/6.0)*b_3_0 + (1.0/6.0)*b_3_1*s + (1.0/6.0)*b_3_2*x3 + (1.0/6.0)*b_3_3*x5 + (1.0/6.0)*b_3_4*x10) - x15*x7 + y*(a_1_0 + a_1_1*s + a_1_2*x3 + a_1_3*x5 + a_1_4*x10 + a_2_0*x + a_2_1*x2 + a_2_2*x12 + a_2_3*x13 + a_2_4*x16 + a_3_0*x8 + a_3_1*x9 + a_3_2*x3*x8 + a_3_3*x5*x8 + a_3_4*x10*x8);
}
}