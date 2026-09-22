# Third-party notices

`qsm-core` is distributed under the MIT licence (see `LICENSE`). Some modules were
written with reference to third-party source that carries its own licence. Those
licences are reproduced below, and the modules concerned name them in their
documentation.

No third-party source is vendored into this repository.

---

## scikit-image — `skimage/restoration/unwrap_3d_ljmu.c`

Consulted while implementing `src/unwrap/bestpath.rs` (best-path / 3D-SRNCP phase
unwrapping).

The reference implementation of the algorithm was written by Hussein Abdul-Rahman
and Munther Gdeisat at Liverpool John Moores University, and adapted for
Python/scikit-image by Gregor Thalhammer. scikit-image distributes it under the
BSD-3-Clause licence.

The underlying algorithm is published in:

- H. Abdul-Rahman, M. A. Gdeisat, D. R. Burton and M. J. Lalor, "Fast
  three-dimensional phase-unwrapping algorithm based on sorting by reliability
  following a non-continuous path", *Proc. SPIE* 5856 (2005) 32–40.
  <https://doi.org/10.1364/AO.46.006623>

```
Copyright (C) 2009-2022 the scikit-image team
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

 1. Redistributions of source code must retain the above copyright notice,
    this list of conditions and the following disclaimer.
 2. Redistributions in binary form must reproduce the above copyright notice,
    this list of conditions and the following disclaimer in the documentation
    and/or other materials provided with the distribution.
 3. Neither the name of skimage nor the names of its contributors may be used
    to endorse or promote products derived from this software without specific
    prior written permission.

THIS SOFTWARE IS PROVIDED BY THE AUTHOR ``AS IS'' AND ANY EXPRESS OR IMPLIED
WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF
MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO
EVENT SHALL THE AUTHOR BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR
BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER
IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
POSSIBILITY OF SUCH DAMAGE.
```
