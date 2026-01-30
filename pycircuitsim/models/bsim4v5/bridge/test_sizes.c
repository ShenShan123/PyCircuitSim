#include <stdio.h>
#include "bsim4_iv_core.h"
#include "bsim4_standalone.h"

int main() {
    printf("Structure sizes:\n");
    printf("  sizeof(BSIM4_Model) = %zu bytes\n", sizeof(BSIM4_Model));
    printf("  sizeof(BSIM4_Instance) = %zu bytes\n", sizeof(BSIM4_Instance));
    printf("  sizeof(BSIM4_Internal) = %zu bytes\n", sizeof(BSIM4_Internal));
    printf("  sizeof(BSIM4_Output) = %zu bytes\n", sizeof(BSIM4_Output));
    printf("  sizeof(BSIM4_States) = %zu bytes\n", sizeof(BSIM4_States));

    printf("\nExpected offsets in BSIM4_Internal:\n");
    printf("  Vth: %zu\n", __builtin_offsetof(BSIM4_Internal, Vth));
    printf("  Ids: %zu\n", __builtin_offsetof(BSIM4_Internal, Ids));
    printf("  Gm: %zu\n", __builtin_offsetof(BSIM4_Internal, Gm));

    return 0;
}
