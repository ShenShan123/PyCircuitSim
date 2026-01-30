#include <stdio.h>
#include <string.h>
#include "bsim4_iv_core.h"

// Declare the core evaluation function
extern int bsim4_iv_evaluate(
    const BSIM4_Model *model,
    const BSIM4_Instance *instance,
    double Vds,
    double Vgs,
    double Vbs,
    BSIM4_Internal *i);

// Simple initialization without all the freePDK45 parameters
void BSIM4_InitModel_45nm_NMOS_SIMPLE(BSIM4_Model *model) {
    memset(model, 0, sizeof(BSIM4_Model));

    model->type = 1;  // NMOS
    model->mobMod = 0;

    // Essential parameters only
    model->tox = 1.14e-9;
    model->toxp = 1.0e-9;
    model->toxm = 1.14e-9;
    model->epsrox = 3.9;
    model->coxe = model->epsrox * 8.854e-12 / model->tox;  // EPS0 / tox

    model->vth0 = 0.322;
    model->k1 = 0.4;
    model->k2 = 0.0;
    model->k3 = 0.0;
    model->k3b = 0.0;

    model->u0 = 0.045;
    model->vsat = 148000.0;

    model->nfactor = 2.1;
    model->ndep = 3.4e18;
    model->nsd = 2.0e20;

    model->phin = 0.0;  // Surface potential
    model->temp = 300.0;

    model->tnom = 27.0;
}

int main() {
    BSIM4_Model model;
    BSIM4_Instance instance;
    BSIM4_Internal internal;

    BSIM4_InitModel_45nm_NMOS_SIMPLE(&model);
    BSIM4_InitInstance(&instance, 45e-9, 90e-9);

    double Vds = 0.1, Vgs = 0.5, Vbs = 0.0;
    int ret = bsim4_iv_evaluate(&model, &instance, Vds, Vgs, Vbs, &internal);

    printf("Simple initialization test:\n");
    printf("Vth = %.6f V\n", internal.Vth);
    printf("Ids = %.6e A = %.3f µA\n", internal.Ids, internal.Ids*1e6);

    return 0;
}
