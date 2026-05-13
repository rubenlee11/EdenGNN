subroutine cubic_spline_eval(n, x, idx, r, a, b, c, d, y)
    implicit none
    integer, intent(in) :: n
    real(8), intent(in) :: x(n)
    integer, intent(in) :: idx(n)
    real(8), intent(in) :: r(:)
    real(8), intent(in) :: a(:), b(:), c(:), d(:)
    real(8), intent(out) :: y(n)

    integer :: i, k
    real(8) :: dx

    do i = 1, n
        k = idx(i)
        dx = x(i) - r(k)
        y(i) = a(k) + b(k)*dx + c(k)*dx*dx + d(k)*dx*dx*dx
    end do

end subroutine