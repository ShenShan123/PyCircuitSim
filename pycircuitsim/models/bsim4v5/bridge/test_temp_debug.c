#include <stdio.h>
#include "bsim4_iv_core.h"

int main() {
    BSIM4_Model model;
    BSIM4_Instance instance;
    BSIM4_Internal internal;

    /* Initialize NMOS model */
    BSIM4_InitModel_45nm_NMOS(&model);
    BSIM4_InitInstance(&instance, 45e-9, 90e-9);

    /* Check temperature parameters */
    printf("Temperature parameters:\n");
    printf("  tnom = %.1f K\n", model.tnom);
    printf("  kt1 = %.6f V-m/K\n", model.kt1);
    printf("  kt2 = %.6f V/K\n", model.kt2);
    printf("  ute = %.2f\n", model.ute);
    printf("  at = %.2f\n", model.at);
    printf("  u0 = %.6f m^2/V-s\n\n", model.u0);

    /* Test at two temperatures */
    double temps[] = {300, 400};
    for (int i = 0; i < 2; i++) {
        model.temp = temps[i];
        bsim4_iv_evaluate(&model, &instance, 0.1, 0.5, 0.0, &internal);

        printf("T = %.0f K:\n", temps[i]);
        printf("  Vth = %.6f V\n", internal.Vth);
        printf("  Ids = %.6e A\n", internal.Ids);
        printf("  ueff = %.6e m^2/V-s\n", internal.ueff);
        printf("  Vgsteff = %.6f V\n\n", internal.Vgsteff);
    }

    return 0;
}
