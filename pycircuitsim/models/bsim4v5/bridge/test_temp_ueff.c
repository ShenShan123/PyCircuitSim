#include <stdio.h>
#include "bsim4_iv_core.h"

int main() {
    BSIM4_Model model;
    BSIM4_Instance instance;
    BSIM4_Internal internal;

    /* Initialize NMOS model */
    BSIM4_InitModel_45nm_NMOS(&model);
    BSIM4_InitInstance(&instance, 45e-9, 90e-9);

    /* Test at room temperature */
    model.temp = 300.0;
    printf("T = 300 K:\n");
    printf("  u0 = %.6e m^2/V-s\n", model.u0);
    printf("  ute = %.2f\n", model.ute);

    int ret = bsim4_iv_evaluate(&model, &instance, 0.1, 0.5, 0.0, &internal);
    printf("  Return code: %d\n", ret);
    printf("  ueff = %.6e m^2/V-s\n", internal.ueff);
    printf("  Vth = %.6f V\n", internal.Vth);
    printf("  Ids = %.6e A\n\n", internal.Ids);

    /* Test at high temperature */
    model.temp = 400.0;
    printf("T = 400 K:\n");
    ret = bsim4_iv_evaluate(&model, &instance, 0.1, 0.5, 0.0, &internal);
    printf("  Return code: %d\n", ret);
    printf("  ueff = %.6e m^2/V-s\n", internal.ueff);
    printf("  Vth = %.6f V\n", internal.Vth);
    printf("  Ids = %.6e A\n", internal.Ids);

    return 0;
}
