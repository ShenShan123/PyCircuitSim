#include <stdio.h>
#include "bsim4_iv_core.h"

int main() {
    BSIM4_Model model;
    BSIM4_Instance instance;
    BSIM4_Internal internal;

    BSIM4_InitModel_45nm_NMOS(&model);
    BSIM4_InitInstance(&instance, 45e-9, 90e-9);

    printf("Model parameters:\n");
    printf("  type = %d\n", model.type);
    printf("  vth0 = %.6f V\n", model.vth0);
    printf("  k1 = %.6f\n", model.k1);
    printf("  k2 = %.6f\n", model.k2);
    printf("  k3 = %.6f\n", model.k3);
    printf("  k3b = %.6f\n", model.k3b);
    printf("  nfactor = %.6f\n", model.nfactor);
    printf("  ndep = %.6e\n", model.ndep);
    printf("  phin = %.6f V\n", model.phin);
    printf("  voff = %.6f V\n", model.voff);
    printf("  tox = %.6e m\n", model.tox);
    printf("  epsrox = %.6f\n", model.epsrox);
    printf("  coxe = %.6e F\n", model.coxe);
    printf("  sqrtPhi = %.6e\n", model.sqrtPhi);
    printf("  Xdep0 = %.6e m\n", model.Xdep0);
    printf("  cdep0 = %.6e F/m^2\n", model.cdep0);
    printf("\n");

    double Vds = 0.1, Vgs = 0.5, Vbs = 0.0;
    int ret = bsim4_iv_evaluate(&model, &instance, Vds, Vgs, Vbs, &internal);

    printf("Results:\n");
    printf("  Vgs = %.3f V\n", Vgs);
    printf("  Vth = %.6f V\n", internal.Vth);
    printf("  Vgsteff = %.6f V\n", internal.Vgsteff);
    printf("  Ids = %.6e A = %.3f µA\n", internal.Ids, internal.Ids*1e6);

    return 0;
}
