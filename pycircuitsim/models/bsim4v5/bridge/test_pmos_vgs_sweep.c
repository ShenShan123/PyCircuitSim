#include <stdio.h>
#include <math.h>
#include "bsim4_iv_core.h"

#define Q_CHARGE 1.60219e-19
#define K_BOLTZMANN 1.380649e-23

int main() {
    BSIM4_Model model;
    BSIM4_Instance instance;
    BSIM4_Internal internal;

    /* Initialize PMOS model */
    BSIM4_InitModel_45nm_PMOS(&model);
    BSIM4_InitInstance(&instance, 45e-9, 90e-9);

    printf("PMOS Vgs Sweep Test (L=45nm, W=90nm, Vds=0.1V)\n");
    printf("==============================================\n\n");

    printf("Vth = %.4f V (from model)\n", model.vth0 * model.type);

    printf("\n%10s | %12s | %12s\n", "Vgs (V)", "Ids (µA)", "Vgsteff (V)");
    printf("----------|--------------|--------------\n");

    /* Test various gate voltages */
    double Vds = 0.1;
    double Vgs_values[] = {-0.3, -0.4, -0.5, -0.6, -0.7, -0.8, -1.0};

    for (int i = 0; i < 7; i++) {
        double Vgs = Vgs_values[i];
        int ret = bsim4_iv_evaluate(&model, &instance, Vds, Vgs, 0.0, &internal);

        printf("%10.2f | %12.3f | %12.6f\n",
               Vgs, internal.Ids * 1e6, internal.Vgsteff);
    }

    return 0;
}
