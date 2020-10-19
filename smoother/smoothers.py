def bspline(fd, n_basis, order):
    from skfda.representation.basis import BSpline

    basis = BSpline(n_basis=n_basis, order=order)
    basis_fd = fd.to_basis(basis)

    return basis_fd