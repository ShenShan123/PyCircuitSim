#include <stdio.h>
#include "bsim4_iv_core.h"

int main() {
    BSIM4_Model model;
    BSIM4_Instance instance;
    BSIM4_Internal internal;
    
    BSIM4_InitModel_45nm_NMOS(&model);
    BSIM4_InitInstance(&instance, 45e-9, 90e-9);
    
    double Vds = 0.1, Vgs = 0.5, Vbs = 0.0;
    bsim4_iv_evaluate(&model, &instance, Vds, Vgs, Vbs, &internal);
    
    printf("Vgs = %.3f V\n", Vgs);
    printf("Vth = %.6f V\n", internal.Vth);
    printf("Vgsteff = %.6f V\n", internal.Vgsteff);
    printf("Ids = %.6e A\n", internal.Ids);
    
    return 0;
}
