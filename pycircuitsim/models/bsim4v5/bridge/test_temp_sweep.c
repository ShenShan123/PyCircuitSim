#include <stdio.h>
#include "bsim4_iv_core.h"

int main() {
    BSIM4_Model model;
    BSIM4_Instance instance;
    BSIM4_Internal internal;

    /* Initialize NMOS model */
    BSIM4_InitModel_45nm_NMOS(&model);
    BSIM4_InitInstance(&instance, 45e-9, 90e-9);

    /* Bias point */
    double Vds = 0.1, Vgs = 0.5, Vbs = 0.0;

    /* Test at different temperatures */
    printf("Temperature Effects Test for NMOS (L=45nm, W=90nm)\n");
    printf("=====================================================\n");
    printf("Vgs = %.2f V, Vds = %.2f V, Vbs = %.2f V\n\n", Vgs, Vds, Vbs);
    printf("Temp (K) | Vth (V) | Id (µA) | ueff (cm²/V-s)\n");
    printf("---------|---------|---------|----------------\n");

    double temps[] = {200, 250, 300, 350, 400, 450};
    int ntemps = sizeof(temps) / sizeof(temps[0]);

    for (int i = 0; i < ntemps; i++) {
        model.temp = temps[i];
        bsim4_iv_evaluate(&model, &instance, Vds, Vgs, Vbs, &internal);

        printf("%8.0f | %7.4f | %7.2f | %14.2f\n",
               temps[i], internal.Vth, internal.Ids * 1e6, internal.ueff * 1e4);
    }

    printf("\nExpected trends:\n");
    printf("- Vth should decrease with temperature (typical: -0.5 to -1.5 mV/K)\n");
    printf("- Id should decrease with temperature (for MOSFETs in saturation)\n");
    printf("- ueff should decrease with temperature (T^ute, ute<0)\n");

    return 0;
}
