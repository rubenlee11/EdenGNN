!
! python -m numpy.f2py -c io_chgcar.f90 -m io_chgcar
!
module io_chgcar
    implicit none
contains
    subroutine get_chgcar_info(filename, natoms, nx, ny, nz, naug, ntypes)
        character(len=*), intent(in) :: filename
        integer, intent(out) :: natoms, nx, ny, nz, naug, ntypes
        
        integer :: u, i, ios
        character(len=256) :: line
        integer :: type_counts(100)
        integer :: n_grid_pts, n_density_lines
        
        open(newunit=u, file=trim(filename), status='old', action='read')
        
        ! Skip title, scaling factor, and lattice vectors (5 lines)
        do i = 1, 5
            read(u, '(A)') line
        end do
        
        ! Skip element symbols (Line 6)
        read(u, '(A)') line
        
        ! Read atom counts (Line 7)
        read(u, '(A)') line
        type_counts = 0
        read(line, *, iostat=ios) type_counts
        
        ! Calculate natoms and ntypes
        natoms = 0
        ntypes = 0
        do i = 1, 100
            if (type_counts(i) > 0) then
                ntypes = ntypes + 1
                natoms = natoms + type_counts(i)
            else
                exit
            end if
        end do
        
        ! Skip coordinate type (Direct/Cartesian)
        read(u, '(A)') line
        
        ! Skip atomic positions
        do i = 1, natoms
            read(u, '(A)') line
        end do
        
        ! Skip empty line
        read(u, '(A)') line
        
        ! Read grid dimensions
        read(u, *) nx, ny, nz
        
        ! Skip density data
        n_grid_pts = nx * ny * nz
        n_density_lines = (n_grid_pts + 4) / 5
        do i = 1, n_density_lines
            read(u, '(A)') line
        end do
        
        ! Count augmentation lines
        naug = 0
        do
            read(u, '(A)', iostat=ios) line
            if (ios /= 0) exit
            naug = naug + 1
        end do
        
        close(u)
    end subroutine get_chgcar_info

    subroutine read_chgcar(filename, natoms, nx, ny, nz, naug, ntypes, &
                           atom_counts, elements_str, cell, pos, density, aug_str)
        character(len=*), intent(in) :: filename
        integer, intent(in) :: natoms, nx, ny, nz, naug, ntypes
        
        integer, intent(out) :: atom_counts(ntypes)
        character(len=256), intent(out) :: elements_str
        real(8), intent(out) :: cell(3, 3)
        real(8), intent(out) :: pos(3, natoms)
        real(8), intent(out) :: density(nx * ny * nz)
        character(len=256), intent(out) :: aug_str(naug)
        
        integer :: u, i
        character(len=256) :: line
        real(8) :: scale
        
        open(newunit=u, file=trim(filename), status='old', action='read')
        
        ! Read title
        read(u, '(A)') line
        
        ! Read scaling factor
        read(u, *) scale
        
        ! Read lattice vectors
        do i = 1, 3
            read(u, *) cell(1, i), cell(2, i), cell(3, i)
            cell(:, i) = cell(:, i) * scale
        end do
        
        ! Read element symbols (Line 6)
        read(u, '(A)') elements_str
        
        ! Read atom counts (Line 7)
        read(u, *) atom_counts
        
        ! Skip coordinate type
        read(u, '(A)') line
        
        ! Read atomic positions
        do i = 1, natoms
            read(u, *) pos(1, i), pos(2, i), pos(3, i)
        end do
        
        ! Skip empty line
        read(u, '(A)') line
        
        ! Skip grid dimensions
        read(u, '(A)') line
        
        ! Read density data
        read(u, *) density
        
        ! Read augmentation occupancies strings
        do i = 1, naug
            read(u, '(A)') aug_str(i)
        end do
        
        close(u)
    end subroutine read_chgcar

    subroutine write_density(density, n_points, filename)
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
end module io_chgcar