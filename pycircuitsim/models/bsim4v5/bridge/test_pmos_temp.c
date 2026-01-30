#include <stdio.h>
#include "bsim4_iv_core.h"

int main() {
    BSIM4_Model model;
    BSIM4_Instance instance;
    BSIM4_Internal internal;

    /* Initialize PMOS model */
    BSIM4_InitModel_45nm_PMOS(&model);
    BSIM4_InitInstance(&instance, 45e-9, 90e-9);

    printf("Testing PMOS at T=300K, Vds=0.1V\n");
    model.temp = 300.0;
    printf("  model->type = %d\n", model.type);
    printf("  model->vth0 = %.4f V\n", model.vth0);
    printf("  model->u0 = %.6e m^2/V-s\n", model.u0);

    /* Test with negative Vgs (for PMOS) */
    int ret = bsim4_iv_evaluate(&model, &instance, 0.1, -0.5, 0.0, &internal);
    printf("  Return code: %d\n", ret);
    printf("  Ids = %.6e A\n", internal.Ids);
    printf("  Vth = %.6f V\n", internal.Vth);
    printf("  Vgsteff = %.6f V\n", internal.Vgsteff);

    return 0;
}
