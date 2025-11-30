import numpy as np
from scipy import ndimage, signal

# === Deconvolution with gradient descent ===

def dotIm(im1, im2):
    """Sum of elementwise products over all channels and pixels (scalar)."""
    im1 = np.asarray(im1, dtype=np.float64)
    im2 = np.asarray(im2, dtype=np.float64)
    return float(np.sum(im1 * im2))


def applyKernel(im, kernel):
    ''' return Mx, where x is im (per-channel convolution) '''
    im = np.asarray(im, dtype=np.float64)
    # support single-channel (H,W) and color (H,W,C)
    if im.ndim == 2:
        return ndimage.convolve(im, kernel, mode='reflect')
    out = np.zeros_like(im, dtype=np.float64)
    for c in range(im.shape[2]):
        out[..., c] = ndimage.convolve(im[..., c], kernel, mode='reflect')
    return out


def applyConjugatedKernel(im, kernel):
    ''' return M^T x, where x is im
        For convolutional M, the adjoint is convolution with flipped kernel.
    '''
    flipped = np.flipud(np.fliplr(kernel))
    return applyKernel(im, flipped)


def computeResidual(kernel, x, y):
    ''' return y - Mx '''
    return y - applyKernel(x, kernel)


def computeStepSize(r, kernel):
    ''' alpha = (r·r) / (r · (A r)), A = M^T M '''
    Ar = applyConjugatedKernel(applyKernel(r, kernel), kernel)
    num = dotIm(r, r)
    den = dotIm(r, Ar)
    if den == 0:
        return 0.0
    return float(num / den)


def deconvGradDescent(im_blur, kernel, niter=10, verbose=False):
    ''' return deblurred image using gradient descent '''
    y = np.asarray(im_blur, dtype=np.float64)
    x = np.zeros_like(y, dtype=np.float64)  # start from black
    for i in range(niter):
        # r = M^T (y - M x)
        r = applyConjugatedKernel(y - applyKernel(x, kernel), kernel)
        alpha = computeStepSize(r, kernel)
        if alpha == 0:
            break
        x = x + alpha * r
        if verbose:
            energy = dotIm(applyKernel(x, kernel) - y, applyKernel(x, kernel) - y)
            print(f"[GD] iter {i+1}/{niter} energy={energy:.6e} alpha={alpha:.6e}")
    return x


# === Deconvolution with conjugate gradient ===

def computeGradientStepSize(r, d, kernel):
    ''' alpha = (r·r) / (d · A d) where A = M^T M '''
    Ad = applyConjugatedKernel(applyKernel(d, kernel), kernel)
    num = dotIm(r, r)
    den = dotIm(d, Ad)
    if den == 0:
        return 0.0
    return float(num / den)


def computeConjugateDirectionStepSize(old_r, new_r):
    ''' beta = (new_r·new_r) / (old_r·old_r) '''
    old_norm = dotIm(old_r, old_r)
    if old_norm == 0:
        return 0.0
    return float(dotIm(new_r, new_r) / old_norm)


def deconvCG(im_blur, kernel, niter=10, verbose=False):
    ''' return deblurred image using conjugate gradient for M^T M x = M^T y '''
    y = np.asarray(im_blur, dtype=np.float64)
    b = applyConjugatedKernel(y, kernel)  # b = M^T y
    x = np.zeros_like(b, dtype=np.float64)
    # A operator: A(v) = M^T (M v)
    def A(v):
        return applyConjugatedKernel(applyKernel(v, kernel), kernel)
    r = b - A(x)
    d = r.copy()
    rr_old = dotIm(r, r)
    for i in range(niter):
        Ad = A(d)
        denom = dotIm(d, Ad)
        if denom == 0:
            break
        alpha = rr_old / denom
        x = x + alpha * d
        r = r - alpha * Ad
        rr_new = dotIm(r, r)
        if rr_old == 0:
            break
        beta = rr_new / rr_old
        d = r + beta * d
        rr_old = rr_new
        if verbose:
            energy = dotIm(applyKernel(x, kernel) - y, applyKernel(x, kernel) - y)
            print(f"[CG] iter {i+1}/{niter} energy={energy:.6e} alpha={alpha:.6e} beta={beta:.6e}")
    return x


def laplacianKernel():
    ''' a 3-by-3 Laplacian kernel '''
    return np.array([[0.0, -1.0, 0.0],
                     [-1.0, 4.0, -1.0],
                     [0.0, -1.0, 0.0]], dtype=np.float64)


def applyLaplacian(im):
    ''' return Lx (x is im), applied per-channel '''
    return applyKernel(im, laplacianKernel())


def applyAMatrix(im, kernel):
    ''' return Ax, where A = M^T M '''
    return applyConjugatedKernel(applyKernel(im, kernel), kernel)


def applyRegularizedOperator(im, kernel, lamb):
    ''' (A + lambda L ) x '''
    return applyAMatrix(im, kernel) + lamb * applyLaplacian(im)


def computeGradientStepSize_reg(grad, p, kernel, lamb):
    ''' alpha = (grad·grad) / (p · (A + lambda L) p) '''
    Ap = applyRegularizedOperator(p, kernel, lamb)
    num = dotIm(grad, grad)
    den = dotIm(p, Ap)
    if den == 0:
        return 0.0
    return float(num / den)


def deconvCG_reg(im_blur, kernel, lamb=0.05, niter=10, verbose=False):
    ''' return deblurred and regularized image using CG on (A + lambda L) x = b '''
    y = np.asarray(im_blur, dtype=np.float64)
    b = applyConjugatedKernel(y, kernel)
    x = np.zeros_like(b, dtype=np.float64)
    r = b - applyRegularizedOperator(x, kernel, lamb)
    d = r.copy()
    rr_old = dotIm(r, r)
    for i in range(niter):
        Ad = applyRegularizedOperator(d, kernel, lamb)
        denom = dotIm(d, Ad)
        if denom == 0:
            break
        alpha = rr_old / denom
        x = x + alpha * d
        r = r - alpha * Ad
        rr_new = dotIm(r, r)
        if rr_old == 0:
            break
        beta = rr_new / rr_old
        d = r + beta * d
        rr_old = rr_new
        if verbose:
            energy = dotIm(applyKernel(x, kernel) - y, applyKernel(x, kernel) - y) + lamb * dotIm(x, applyLaplacian(x))
            print(f"[CG-reg] iter {i+1}/{niter} energy={energy:.6e} alpha={alpha:.6e} beta={beta:.6e}")
    return x


def naiveComposite(bg, fg, mask, y, x):
    ''' naive composition: copy fg into bg at (y,x) where mask==1 (vectorized) '''
    bg = np.asarray(bg, dtype=np.float64).copy()
    fg = np.asarray(fg, dtype=np.float64)
    h, w = fg.shape[:2]

    # prepare mask with channels
    if fg.ndim == 3:
        channels = fg.shape[2]
    else:
        channels = 1
        fg = fg[..., None]

    if mask.ndim == 2:
        m3 = np.repeat(mask[..., None], channels, axis=2)
    else:
        m3 = mask

    # slice target region
    bg_patch = bg[y:y+h, x:x+w]
    if bg_patch.shape[0] != h or bg_patch.shape[1] != w:
        raise ValueError("fg+offset does not fit inside bg")

    # Correct blending logic
    out_patch = bg_patch * (1 - m3) + fg * m3

    bg[y:y+h, x:x+w] = out_patch
    return bg


def Poisson(bg, fg, mask, niter=200, verbose=False):
    ''' Poisson editing using gradient descent (masked).
        bg, fg: HxWxC, mask: HxW (0/1). '''
    bg = np.asarray(bg, dtype=np.float64)
    fg = np.asarray(fg, dtype=np.float64)
    mask = np.asarray(mask, dtype=np.float64)
    if bg.shape != fg.shape:
        raise ValueError("bg and fg must have same shape for this implementation")
    H, W = mask.shape
    # b = Laplacian(fg)
    b = applyLaplacian(fg)
    # initialize x: outside mask use bg; inside mask start at 0
    x = bg.copy()
    x[mask == 1] = 0.0
    for i in range(niter):
        Ax = applyLaplacian(x)
        r = b - Ax
        # updates happen only inside mask
        r = r * mask[..., None]
        Ar = applyLaplacian(r) * mask[..., None]
        num = dotIm(r, r)
        den = dotIm(r, Ar)
        if den == 0:
            break
        alpha = num / den
        x = x + alpha * r * mask[..., None]
    # enforce outside mask stays bg
    x[mask == 0] = bg[mask == 0]
    if verbose and (i % max(1, niter//10) == 0):
        print(f"[Poisson-GD] iter {i+1}/{niter} residual_norm={np.sqrt(num):.6e} alpha={alpha:.6e}")
    return x


def PoissonCG(bg, fg, mask, niter=200, verbose=False):
    ''' Poisson editing using conjugate gradient (masked).
        bg, fg: HxWxC, mask: HxW (0/1). '''
    bg = np.asarray(bg, dtype=np.float64)
    fg = np.asarray(fg, dtype=np.float64)
    mask = np.asarray(mask, dtype=np.float64)
    if bg.shape != fg.shape:
        raise ValueError("bg and fg must have same shape for this implementation")
    # b = Laplacian(fg) but zeroed outside mask
    b = applyLaplacian(fg) * mask[..., None]

    # initialize x: bg values outside mask, zeros inside (unknowns)
    x = bg.copy()
    x[mask == 1] = 0.0

    # A operator restricted to mask: A(v) = (Laplacian(v)) * mask
    def Aop(v):
        return applyLaplacian(v) * mask[..., None]

    Ax = Aop(x)
    r = b - Ax
    d = r.copy()
    rr_old = dotIm(r, r)
    for i in range(niter):
        Ad = Aop(d)
        denom = dotIm(d, Ad)
        if denom == 0:
            break
        alpha = rr_old / denom
        x = x + alpha * d * mask[..., None]
    # enforce outside mask stays bg
        x[mask == 0] = bg[mask == 0]
        r = r - alpha * Ad
        rr_new = dotIm(r, r)
        if rr_old == 0:
            break
        beta = rr_new / rr_old
        d = r + beta * d
        rr_old = rr_new
        if verbose and (i % max(1, niter//10) == 0):
            print(f"[Poisson-CG] iter {i+1}/{niter} residual={np.sqrt(rr_new):.6e} alpha={alpha:.6e} beta={beta:.6e}")
    x[mask == 0] = bg[mask == 0]
    return x





# Wrapper with lowercase name to match assignment spec
def poisson(bg, fg, mask, niter=200):
    return Poisson(bg, fg, mask, niter=niter)


#==== Helpers. Use them as possible. ==== 

def convolve3(im, kernel):
  from scipy import ndimage
  center=(0,0)
  r=ndimage.filters.convolve(im[:,:,0], kernel, mode='reflect', origin=center) 
  g=ndimage.filters.convolve(im[:,:,1], kernel, mode='reflect', origin=center) 
  b=ndimage.filters.convolve(im[:,:,2], kernel, mode='reflect', origin=center) 
  return (np.dstack([r,g,b]))

def gauss2D(sigma=2, truncate=3):
  kernel=horiGaussKernel(sigma, truncate);
  kerker=np.dot(kernel.transpose(), kernel)
  return kerker/sum(kerker.flatten())

def horiGaussKernel(sigma, truncate=3):
  from scipy import signal
  sig=signal.gaussian(2*int(sigma*truncate)+1,sigma)
  return np.array([sig/sum(sig)])