# Native FLIMLib backend

`flimfit_labview.dll` is a 64-bit Windows C DLL rebuilt from the supplied
`flimfit_labview.c` wrapper and FLIMLib 2.2.6-SNAPSHOT source. The wrapper uses
the C `__cdecl` ABI and the Python bridge calls only `flim_fit_pixels_v2` and
`flim_fit_global_pixels_v2`.

Corrections applied before rebuilding:

- supply the sigma buffer required by FLIMLib's constant-noise mode;
- convert the wrapper's inclusive stop bin to FLIMLib's exclusive end;
- report successful training pixels as status 1;
- return status -6 for an underdetermined independent fit window.

The binary SHA-256 is
`067f7e338d780cdd1e078de6544a15791ec0321159d6cc6d9c6f02eab4327ffb`.

## Rebuild

On a Windows machine with CMake and MSVC installed:

```powershell
cmake -S native -B native/build -G "NMake Makefiles" -DCMAKE_BUILD_TYPE=Release
cmake --build native/build
```

The exact linked FLIMLib source subset is in `flimlib_source/src/main/c`.
FLIMLib is GPL-3.0-or-later; see `FLIMLib_LICENSE_GPL-3.0.txt` and
`flimlib_source/LICENSE.txt`.

Do not replace this DLL with the original unpatched build while enabling
constant weighting. Native fitting should remain serialized, and untrusted
replacement DLLs should be tested out of process before use.
