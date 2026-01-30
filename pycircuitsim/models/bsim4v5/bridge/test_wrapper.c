#include <stdio.h>
#include "bsim4_standalone.h"

int main() {
    BSIM4_Model model;
    BSIM4_Instance instance;
    BSIM4_Output output;

    BSIM4_InitModel_45nm_NMOS(&model);
    BSIM4_InitInstance(&instance, 45e-9, 90e-9);

    double Vds = 0.1, Vgs = 0.5, Vbs = 0.0;
    int ret = BSIM4_Evaluate(&model, &instance, Vds, Vgs, Vbs, &output);

    printf("BSIM4_Evaluate wrapper test:\n");
    printf("Vgs = %.3f V\n", Vgs);
    printf("Vth = %.6f V\n", output.Vth);
    printf("Vgsteff = %.6f V\n", output.Vgsteff);
    printf("Id = %.6e A = %.3f µA\n", output.Id, output.Id*1e6);
    printf("Gm = %.6e S = %.3f µS\n", output.Gm, output.Gm*1e6);
    printf("Gds = %.6e S = %.3f µS\n", output.Gds, output.Gds*1e6);

    return 0;
}
