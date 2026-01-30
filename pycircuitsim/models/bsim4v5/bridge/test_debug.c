#include <stdio.h>
#include "bsim4_iv_core.h"

int main() {
    BSIM4_Model model;
    BSIM4_Instance instance;
    BSIM4_Internal internal;

    printf("Before initialization:\n");
    printf("  instance.L = %e\n", instance.L);
    printf("  instance.W = %e\n", instance.W);

    BSIM4_InitModel_45nm_NMOS(&model);
    BSIM4_InitInstance(&instance, 45e-9, 90e-9);

    printf("\nAfter initialization:\n");
    printf("  instance.L = %e m\n", instance.L);
    printf("  instance.W = %e m\n", instance.W);
    printf("  model.vth0 = %e V\n", model.vth0);
    printf("  model.u0 = %e\n", model.u0);
    printf("  model.tox = %e\n", model.tox);

    double Vds = 0.1, Vgs = 0.5, Vbs = 0.0;

    printf("\nCalling bsim4_iv_evaluate...\n");
    int ret = bsim4_iv_evaluate(&model, &instance, Vds, Vgs, Vbs, &internal);

    printf("\nResults:\n");
    printf("  Vgs = %.3f V\n", Vgs);
    printf("  Vth = %.6f V\n", internal.Vth);
    printf("  Vgsteff = %.6f V\n", internal.Vgsteff);
    printf("  Ids = %.6e A = %.3f µA\n", internal.Ids, internal.Ids*1e6);

    return 0;
}
