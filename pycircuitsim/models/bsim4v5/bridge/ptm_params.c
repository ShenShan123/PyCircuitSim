#include <stdio.h>
#include <string.h>
#include "bsim4_iv_core.h"

// PTM 45nm NMOS parameters (simplified)
void BSIM4_InitModel_PTM_45nm_NMOS(BSIM4_Model *model) {
    memset(model, 0, sizeof(BSIM4_Model));

    model->type = 1;  // NMOS
    model->mobMod = 0;

    // Oxide parameters
    model->tox = 1.2e-9;
    model->toxp = 1.0e-9;
    model->toxm = 1.2e-9;
    model->epsrox = 3.9;
    model->coxe = model->epsrox * 8.854e-12 / model->tox;

    // Vth parameters - PTM style
    model->vth0 = 0.45;
    model->k1 = 0.4;
    model->k2 = 0.0;
    model->k3 = 0.0;
    model->k3b = 0.0;
    model->voff = 0.0;

    // Mobility
    model->u0 = 0.04;
    model->vsat = 100000.0;

    // Subthreshold
    model->nfactor = 1.0;
    model->ndep = 1.0e18;
    model->nsd = 2.0e20;
    model->phin = 0.0;

    // DIBL and other effects
    model->dvt0 = 1.0;
    model->dvt1 = 2.0;
    model->dvt2 = 0.0;
    model->eta0 = 0.0;
    model->etab = 0.0;

    // CLM
    model->pclm = 0.01;

    // SCBE
    model->pscbe1 = 0.0;
    model->pscbe2 = 0.0;

    // Gate
    model->ngate = 2.5e20;
    model->vfb = -0.7;

    // Parasitic resistance
    model->rdsw = 100.0;
    model->rsw = 50.0;
    model->rdw = 50.0;

    // Geometry corrections
    model->xl = 0.0;
    model->xw = 0.0;

    // Mobility degradation
    model->ua = 1.0e-10;
    model->ub = 1.0e-18;
    model->uc = 0.0;

    // Temperature
    model->temp = 300.0;
    model->tnom = 27.0;
}

int main() {
    BSIM4_Model model;
    BSIM4_Instance instance;
    BSIM4_Internal internal;

    BSIM4_InitModel_PTM_45nm_NMOS(&model);
    BSIM4_InitInstance(&instance, 45e-9, 90e-9);

    printf("PTM 45nm NMOS test:\n");
    printf("  vth0 = %.6f V\n", model.vth0);
    printf("  k1 = %.6f\n", model.k1);
    printf("  voff = %.6f V\n", model.voff);
    printf("\n");

    double Vds = 0.1, Vgs = 0.5, Vbs = 0.0;
    bsim4_iv_evaluate(&model, &instance, Vds, Vgs, Vbs, &internal);

    printf("Results:\n");
    printf("  Vgs = %.3f V\n", Vgs);
    printf("  Vth = %.6f V\n", internal.Vth);
    printf("  Ids = %.6e A = %.3f µA\n", internal.Ids, internal.Ids*1e6);

    return 0;
}
