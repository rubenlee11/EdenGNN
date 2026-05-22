module io_cube
    implicit none
contains

    subroutine get_cube_info(filename, natoms, nx, ny, nz)
        character(len=*), intent(in) :: filename
        integer, intent(out) :: natoms, nx, ny, nz
        
        integer :: u
        character(len=256) :: dummy_line
        real(8) :: dummy_vec(3)

        open(newunit=u, file=trim(filename), status='old', action='read')
        read(u, '(A)') dummy_line
        read(u, '(A)') dummy_line
        read(u, *) natoms, dummy_vec
        read(u, *) nx, dummy_vec
        read(u, *) ny, dummy_vec
        read(u, *) nz, dummy_vec
        close(u)
    end subroutine get_cube_info

    subroutine read_cube_data(filename, natoms, nx, ny, nz, numbers, charges, cell, pos, density)
        character(len=*), intent(in) :: filename
        integer, intent(in) :: natoms, nx, ny, nz
        
        integer, intent(out) :: numbers(natoms)       
        real(8), intent(out) :: charges(natoms)       
        real(8), intent(out) :: cell(3, 3)            
        real(8), intent(out) :: pos(3, natoms)        
        real(8), intent(out) :: density(nx * ny * nz) 

        integer :: u, i
        character(len=256) :: dummy_line
        real(8) :: origin(3), vox_x(3), vox_y(3), vox_z(3)
        integer :: dummy_atnum, dummy_nx, dummy_ny, dummy_nz

        open(newunit=u, file=trim(filename), status='old', action='read')
        
        read(u, '(A)') dummy_line
        read(u, '(A)') dummy_line
        
        read(u, *) dummy_atnum, origin
        read(u, *) dummy_nx, vox_x
        read(u, *) dummy_ny, vox_y
        read(u, *) dummy_nz, vox_z

        cell(1, :) = nx * vox_x
        cell(2, :) = ny * vox_y
        cell(3, :) = nz * vox_z

        do i = 1, natoms
            read(u, *) numbers(i), charges(i), pos(1, i), pos(2, i), pos(3, i)
        end do

        read(u, *) density
        
        close(u)
    end subroutine read_cube_data

    subroutine write_cube_data(filename, natoms, nx, ny, nz, numbers, charges, cell, pos, density)
        character(len=*), intent(in) :: filename
        integer, intent(in) :: natoms, nx, ny, nz
        integer, intent(in) :: numbers(natoms)
        real(8), intent(in) :: charges(natoms)
        real(8), intent(in) :: cell(3, 3)
        real(8), intent(in) :: pos(3, natoms)
        real(8), intent(in) :: density(nx * ny * nz)

        integer :: u, i, ix, iy, idx_start, idx_end
        real(8) :: vox_x(3), vox_y(3), vox_z(3)

        vox_x = cell(1, :) / dble(nx)
        vox_y = cell(2, :) / dble(ny)
        vox_z = cell(3, :) / dble(nz)

        open(newunit=u, file=trim(filename), status='replace', action='write')

        write(u, '(A)') "Cube file created by EdenGNN"
        write(u, '(A)') "OUTER LOOP: X, MIDDLE LOOP: Y, INNER LOOP: Z"
        
        write(u, '(I5, 3F12.6)') natoms, 0.0d0, 0.0d0, 0.0d0
        
        write(u, '(I5, 3F12.6)') nx, vox_x
        write(u, '(I5, 3F12.6)') ny, vox_y
        write(u, '(I5, 3F12.6)') nz, vox_z

        do i = 1, natoms
            write(u, '(I5, 4F12.6)') numbers(i), charges(i), pos(1, i), pos(2, i), pos(3, i)
        end do

        do ix = 1, nx
            do iy = 1, ny
                idx_start = (ix - 1) * ny * nz + (iy - 1) * nz + 1
                idx_end   = idx_start + nz - 1
                write(u, '(6ES18.10)') density(idx_start : idx_end)
            end do
        end do

        close(u)
    end subroutine write_cube_data
end module io_cube