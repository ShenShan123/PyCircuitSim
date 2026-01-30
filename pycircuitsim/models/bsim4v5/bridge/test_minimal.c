#include <stdio.h>
#include "bsim4_iv_core.h"

int main() {
    BSIM4_Model model;
    BSIM4_Instance instance;
    BSIM4_Internal internal;

    // Minimal initialization - only set essential parameters
    memset(&model, 0, sizeof(BSIM4_Model));
    memset(&instance, 0, sizeof(BSIM4_Instance));

    model.type = 1;  // NMOS
    model.tox = 1.14e-9;
    model.vth0 = 0.322;
    model.u0 = 0.045;
    model.vsat = 148000.0;

    instance.L = 45e-9;
    instance.W = 90e-9;

    double Vds = 0.1, Vgs = 0.5, Vbs = 0.0;
    int ret = bsim4_iv_evaluate(&model, &instance, Vds, Vgs, Vbs, &internal);

    printf("Minimal initialization test:\n");
    printf("Vth = %.6f V\n", internal.Vth);
    printf("Ids = %.6e A = %.3f µA\n", internal.Ids, internal.Ids*1e6);

    return 0;
}
