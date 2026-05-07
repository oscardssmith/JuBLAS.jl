# Shared shape list for kernel_sweep / size_sweep / future bench scripts.
# Square sizes step by 8 up to 80 to expose the curve between discrete
# kernel-choice tiers. Rectangular families (skinny C, short C, rank-K)
# cover the same per-axis range so each dim's threshold can be read off
# in one place. Two deep-K cases at the end (M=N small, K large) cover
# that distinct regime.
using Printf

const SHAPES = let
    smalls = 8:8:80
    sq    = [(@sprintf("square-%d",   n), n,    n,    n   ) for n in smalls]
    skin  = [(@sprintf("skinny-N%d",  n), 1024, n,    1024) for n in smalls]
    shrt  = [(@sprintf("short-M%d",   n), n,    1024, 1024) for n in smalls]
    rankK = [(@sprintf("rank-K-K%d",  n), 1024, 1024, n   ) for n in smalls]
    deepK = [("tiny-deep-K",  16, 16, 4096),
             ("tiny-deep-K2", 32, 32, 4096)]
    [sq; skin; shrt; rankK; deepK]
end
