def fit_bkg(tpf: lk.TessTargetPixelFile, polyorder: int = 1) -> np.ndarray:
    """Fit a simple 2d polynomial background to a TPF

    Parameters
    ----------
    tpf: lightkurve.TessTargetPixelFile
        Target pixel file object
    polyorder: int
        Polynomial order for the model fit.

    Returns
    -------
    model : np.ndarray
        Model for background with same shape as tpf.shape
    """
    # Notes for understanding this function
    # All arrays in this func will have dimensions drawn from one of the following: [ntimes, ncols, nrows, npix, ncomp]
    #   ntimes = number of cadences
    #   ncols, nrows = shape of tpf
    #   npix = ncols*nrows, is the length of the unraveled vectors
    #   ncomp = num of components in the polynomial

    # Error catching
    if not isinstance(tpf, lk.TessTargetPixelFile):
        raise ValueError("Input a TESS Target Pixel File")

    if (np.product(tpf.shape[1:]) < 100) | np.any(np.asarray(tpf.shape[1:]) < 6):
        raise ValueError("TPF too small. Use a bigger cut out.")

    # Grid for calculating polynomial
    R, C = np.mgrid[: tpf.shape[1], : tpf.shape[2]].astype(float)
    R -= tpf.shape[1] / 2
    C -= tpf.shape[2] / 2

    # nested b/c we run twice, once on each orbit
    def func(tpf):
        # Design matrix
        A = np.vstack(
            [
                R.ravel() ** idx * C.ravel() ** jdx
                for idx in range(polyorder + 1)
                for jdx in range(polyorder + 1)
            ]
        ).T

        # Median star image
        m = np.median(tpf.flux.value, axis=0)
        # Remove background from median star image
        mask = ~sigma_clip(m, sigma=3).mask.ravel()
        # plt.imshow(mask.reshape(m.shape))
        bkg0 = A.dot(
            np.linalg.solve(A[mask].T.dot(A[mask]), A[mask].T.dot(m.ravel()[mask]))
        ).reshape(m.shape)

        # m is the median frame
        m -= bkg0

        # Include in design matrix
        A = np.hstack([A, m.ravel()[:, None]])

        # Fit model to data, including a model for the stars in the last column
        f = np.vstack(tpf.flux.value.transpose([1, 2, 0]))
        ws = np.linalg.solve(A.T.dot(A), A.T.dot(f))
        # shape of ws is (num of times, num of components)
        # A . ws gives shape (npix, ntimes)

        # Build a model that is just the polynomial
        model = (
            (A[:, :-1].dot(ws[:-1]))
            .reshape((tpf.shape[1], tpf.shape[2], tpf.shape[0]))
            .transpose([2, 0, 1])
        )
        # model += bkg0
        return model

    # Break point for TESS orbit
    # currently selects where the biggest gap in cadences is
    # could cause problems in certain cases with lots of quality masking! Think about how to handle bit masking
    b = np.where(np.diff(tpf.cadenceno) == np.diff(tpf.cadenceno).max())[0][0] + 1

    # Calculate the model for each orbit, then join them
    model = np.vstack([func(tpf) for tpf in [tpf[:b], tpf[b:]]])

    return model