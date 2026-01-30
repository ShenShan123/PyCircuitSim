#include <stdio.h>
#include <math.h>
#include "bsim4_iv_core.h"

/* Physical constants (from bsim4_iv_core.c) */
#define Q_CHARGE 1.60219e-19
#define K_BOLTZMANN 1.380649e-23

int main() {
    BSIM4_Model model;
    BSIM4_Instance instance;
    BSIM4_Internal internal;

    /* Initialize PMOS model */
    BSIM4_InitModel_45nm_PMOS(&model);
    BSIM4_InitInstance(&instance, 45e-9, 90e-9);

    printf("PMOS Debug Test\n");
    printf("===============\n");
    printf("model->type = %d\n", model.type);
    printf("model->vth0 = %.6f V\n", model.vth0);
    printf("model->u0 = %.6e m^2/V-s\n", model.u0);
    printf("model->k1 = %.4f\n", model.k1);
    printf("model->k2 = %.4f\n", model.k2);
    printf("model->tox = %.6e m\n", model.tox);
    printf("model->temp = %.1f K\n", model.temp);
    printf("model->tnom = %.1f K\n", model.tnom);
    printf("model->nsub = %.6e cm^-3\n", model.nsub);

    /* Test with negative Vgs */
    double Vds = 0.1, Vgs = -0.5, Vbs = 0.0;
    printf("\nBias: Vds=%.2f V, Vgs=%.2f V, Vbs=%.2f V\n", Vds, Vgs, Vbs);

    /* Check key calculations */
    double ni = 1.45e16;
    double Vtm = K_BOLTZMANN * model.temp / Q_CHARGE;
    double nsub_si = model.nsub * 1e6;
    double Phis = 2.0 * Vtm * log(nsub_si / ni);
    printf("\nVtm = %.6f V, Phis = %.6f V, sqrt(Phis) = %.6f\n", Vtm, Phis, sqrt(Phis));

    /* Check sqrtPhi calculation */
    double sqrtPhi = 0.0;
    if (model.phin > 0.0) {
        sqrtPhi = sqrt(model.phin);
    } else {
        sqrtPhi = sqrt(Phis);
    }
    printf("sqrtPhi = %.6f\n", sqrtPhi);

    /* Evaluate */
    int ret = bsim4_iv_evaluate(&model, &instance, Vds, Vgs, Vbs, &internal);
    printf("\nReturn code: %d\n", ret);
    printf("Ids = %.6e A\n", internal.Ids);
    printf("Vth = %.6f V\n", internal.Vth);
    printf("Vgsteff = %.6f V\n", internal.Vgsteff);

    return 0;
}
