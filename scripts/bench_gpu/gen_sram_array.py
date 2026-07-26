"""Generate an R x C 6T SRAM array deck (LEVEL=73) for scaling profiles."""
import sys


def sram_array(rows: int, cols: int, vdd=0.75, ln=16, lp=16, nfin=4,
               tech="tsmc5", vt="lvt", analysis=".op") -> str:
    L = [f"* {rows}x{cols} 6T SRAM array — DirectNet ({rows*cols} cells, "
         f"{rows*cols*6} devices)",
         f"Vdd vdd 0 {vdd}"]
    for r in range(rows):
        L.append(f"Vwl{r} wl{r} 0 0.0")
    for c in range(cols):
        L.append(f"Vbl{c} bl{c} 0 {vdd}")
        L.append(f"Vblb{c} blb{c} 0 {vdd}")
    ic = []
    for r in range(rows):
        for c in range(cols):
            q, qb = f"q{r}_{c}", f"qb{r}_{c}"
            L.append(f"X{r}_{c}l {q} {qb} vdd sraminv NF={nfin}")
            L.append(f"X{r}_{c}r {qb} {q} vdd sraminv NF={nfin}")
            L.append(f"Mal{r}_{c} bl{c}  wl{r} {q}  0 nmos_nn L={ln}n NFIN={nfin}")
            L.append(f"Mar{r}_{c} blb{c} wl{r} {qb} 0 nmos_nn L={ln}n NFIN={nfin}")
            ic.append(f"V({q})={vdd} V({qb})=0.0")
    L.append(".ic " + " ".join(ic))
    L += [".subckt sraminv i o vdd NF=1",
          f"Mpl o i vdd vdd pmos_nn L={lp}n NFIN=NF",
          f"Mnl o i 0   0   nmos_nn L={ln}n NFIN=NF",
          ".ends",
          f".model nmos_nn NMOS (LEVEL=73 TECH={tech} VT={vt})",
          f".model pmos_nn PMOS (LEVEL=73 TECH={tech} VT={vt})",
          analysis, ".end"]
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    r, c = int(sys.argv[1]), int(sys.argv[2])
    an = sys.argv[3] if len(sys.argv) > 3 else ".op"
    sys.stdout.write(sram_array(r, c, analysis=an))
