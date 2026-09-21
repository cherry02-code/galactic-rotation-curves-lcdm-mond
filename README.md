# Comparative Analysis of Galactic Rotation Curves vs. MOND

This repository contains the Python implementation used to analyze galactic rotation curves across three representative galaxies from the Spitzer Photometry and Accurate Rotation Curves (SPARC) database: **DDO 154**, **NGC 5055**, and **NGC 3198**.

The project evaluates two competing paradigms in galactic dynamics:
1. **Cold Dark Matter \Lambda$CDM:** Modeling non-baryonic halo velocity contributions in quadrature.
2. **Modified Newtonian Dynamics (MOND):** Applying Milgrom's acceleration modification ($a_0 \approx 1.2 \times 10^{-10} \text{ m/s}^2$) directly to observed baryonic distributions with zero free halo parameters.

---

## 📊 Analyzed Galaxies

| Galaxy | Morphological Type | Acceleration Regime |
| :--- | :--- | :--- |
| **DDO 154** | Gas-dominated dwarf | Deep MOND ($a \ll a_0$) |
| **NGC 5055** | Massive HSB spiral | High acceleration / Bulge-dominated |
| **NGC 3198** | Intermediate spiral | Benchmark transition regime |

---

## 🛠️ Project Setup & Dependencies

Ensure you have Python 3.x installed along with the following libraries:

```bash
pip install numpy matplotlib
