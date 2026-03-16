!
! python -m numpy.f2py -c io_chgcar.f90 -m io_chgcar
!
subroutine write_density(density, n_points, filename)
    !
    ! 
    !
    implicit none
    
    real(8), dimension(*), intent(in) :: density
    integer, intent(in) :: n_points
    character(len=*), intent(in) :: filename
    integer :: i
    integer :: unit
    
    inquire(iolength=unit)
    open(newunit=unit, file=trim(filename), status='unknown', action='write', position='append')

    do i = 1, n_points, 5

        write(unit, '(6(1X,E17.11))') density(i:min(i+4, n_points))
    end do
    close(unit)

end subroutine write_density

subroutine write_aug(aug, filename, natom, nele_atom, nele_total)
    implicit none
    integer, intent(in) :: natom, nele_total
    integer, intent(in) :: nele_atom(natom)
    real(8), intent(in) :: aug(nele_total)
    character(len=*), intent(in) :: filename
    integer ia, i, unit, base
    inquire(iolength=unit)

    open(newunit=unit, file=trim(filename), status='unknown', action='write', position='append')    
    base = 0
    do ia = 1, natom
        write(unit, '(A,I4,I4)') 'augmentation occupancies', ia, nele_atom(ia)
        do i = 1, nele_atom(ia), 5
            write(unit, '(6(1X,E14.7))') aug(i + base : min(i + base + 4, base + nele_atom(ia)))
        end do
        base = base + nele_atom(ia)
    end do
    close(unit)
end subroutine write_aug