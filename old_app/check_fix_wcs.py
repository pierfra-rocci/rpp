from astropy.io import fits
from astropy.wcs import WCS
import re
import sys
import shutil
import numpy as np

DIST_KEYS_PATTERNS = [
    r"^CPDIS",
    r"^DET2IM",
    r"^D?SS",
    r"^PV",
    r"^PVi_",
    r"^PS",  # distortion / pv / lookup hints
    r"^A_",
    r"^B_",
    r"^AP_",
    r"^BP_",
    r"^A_ORDER",
    r"^B_ORDER",  # SIP
    r"^CDELT[M]?",
    r"^CROT",
    r"^CDELTM",  # deprecated / odd keys (we only inspect these)
]


def list_interesting_keys(hdr):
    keys = [
        "CTYPE1",
        "CTYPE2",
        "CRPIX1",
        "CRPIX2",
        "CRVAL1",
        "CRVAL2",
        "CDELT1",
        "CDELT2",
        "CD1_1",
        "CD1_2",
        "CD2_1",
        "CD2_2",
        "CROTA2",
        "WCSAXES",
        "WCSNAME",
        "WCSNAMEa",
    ]
    print("Core WCS keys:")
    for k in keys:
        if k in hdr:
            print(f"  {k:8s} = {hdr[k]}")
    print("\nPotential non-standard / distortion keys found:")
    for k in hdr.keys():
        for pat in DIST_KEYS_PATTERNS:
            if re.match(pat, k):
                print(f"  {k:8s} = {hdr[k]}")
                break


def try_make_wcs(hdul, hdu_idx=0, naxis2=True):
    hdr = hdul[hdu_idx].header
    for kwargs in [
        {"fix": True, "relax": True, "naxis": 2},
        {"fix": True, "relax": True},  # try default axis detection
        {"fix": False, "relax": True, "naxis": 2},
    ]:
        try:
            print(f"\nTrying WCS(...) with {kwargs} ...")
            w = WCS(header=hdr, fobj=hdul, **kwargs)
            print("  SUCCESS: WCS constructed.")
            return w
        except Exception as e:
            print("  FAILED:", type(e).__name__, str(e))
    return None


def remove_distortion_keys_and_write(hdul, outpath, hdu_idx=0):
    hdr = hdul[hdu_idx].header
    to_remove = []
    for k in list(hdr.keys()):
        for pat in DIST_KEYS_PATTERNS:
            if re.match(pat, k):
                to_remove.append(k)
                break
    if not to_remove:
        print("No distortion-like keys found to remove.")
        return False
    print("Removing keys:", to_remove)
    for k in to_remove:
        hdr.pop(k, None)
    # optionally also remove SIP coefficients A_*, B_*, etc. above already matched
    hdul.writeto(outpath, overwrite=True)
    print("Wrote cleaned FITS to", outpath)
    return True


def reconstruct_cd_from_cdelt_crota(hdr):
    # try to construct a CD matrix when CD keywords missing but CDELT/CROTA present.
    if ("CD1_1" in hdr) or ("PC1_1" in hdr):
        print("CD/PC already present; skipping reconstruction.")
        return False
    if "CDELT1" in hdr and "CDELT2" in hdr:
        cdelt1 = float(hdr["CDELT1"])
        cdelt2 = float(hdr["CDELT2"])
        theta = float(hdr.get("CROTA2", 0.0)) * np.pi / 180.0
        # conservative reconstruction (assumes CROTA2 is the rotation)
        cd11 = cdelt1 * np.cos(theta)
        cd12 = -cdelt2 * np.sin(theta)
        cd21 = cdelt1 * np.sin(theta)
        cd22 = cdelt2 * np.cos(theta)
        hdr["CD1_1"] = cd11
        hdr["CD1_2"] = cd12
        hdr["CD2_1"] = cd21
        hdr["CD2_2"] = cd22
        print("Reconstructed CD matrix from CDELT/CROTA2.")
        return True
    return False


def main(path, hdu_idx=0, out_clean=None, remove_distortion=False):
    hdul = fits.open(path)
    print("Opened", path)
    list_interesting_keys(hdul[hdu_idx].header)
    w = try_make_wcs(hdul, hdu_idx=hdu_idx)
    if w is not None:
        print("WCS ready. has_distortion:", w.has_distortion)
        hdul.close()
        return 0
    # try reconstruction
    changed = reconstruct_cd_from_cdelt_crota(hdul[hdu_idx].header)
    if changed:
        w = try_make_wcs(hdul, hdu_idx=hdu_idx)
        if w is not None:
            print("WCS ready after CD reconstruction.")
            if out_clean:
                hdul.writeto(out_clean, overwrite=True)
            hdul.close()
            return 0
    if remove_distortion:
        # create backup then write cleaned header to out_clean
        if not out_clean:
            out_clean = path.replace(".fits", "_clean.fits")
        hdul.close()
        shutil.copy(path, out_clean)
        hdul2 = fits.open(out_clean, mode="update")
        removed = remove_distortion_keys_and_write(hdul2, out_clean, hdu_idx=hdu_idx)
        hdul2.close()
        # re-open and test
        hdul3 = fits.open(out_clean)
        w = try_make_wcs(hdul3, hdu_idx=hdu_idx)
        if w is not None:
            print("WCS ready after removing distortion keywords.")
            hdul3.close()
            return 0
        hdul3.close()
    print(
        "Unable to construct a valid WCS automatically. Inspect keys and try providing the referenced distortion table HDU (if any), or remove broken distortion keywords."
    )
    hdul.close()
    return 2


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(
            "Usage: python check_fix_wcs.py input.fits [--hdu N] [--out cleaned.fits] [--remove-distortion]"
        )
        sys.exit(1)
    path = sys.argv[1]
    hdu = 0
    out = None
    remove = False
    for a in sys.argv[2:]:
        if a.startswith("--hdu"):
            hdu = (
                int(a.split("=")[1])
                if "=" in a
                else int(sys.argv[sys.argv.index(a) + 1])
            )
        if a.startswith("--out"):
            out = a.split("=")[1] if "=" in a else sys.argv[sys.argv.index(a) + 1]
        if a == "--remove-distortion":
            remove = True
    sys.exit(main(path, hdu_idx=hdu, out_clean=out, remove_distortion=remove))
