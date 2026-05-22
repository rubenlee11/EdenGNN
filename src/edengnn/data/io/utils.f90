! python -m numpy.f2py -c -m utils_f utils.f90
module utils
    implicit none
contains
    subroutine set_grid(rcut, grid, ng, gcart, gfrac, nm1, nm2, nm3)
        !---------------------------------------------------------------
        ! description:
        !   find all grids in a sphere: |r| < R (or in reciprocal space, |G| < Ecut)
        ! input variables:
        !   rcut: cutoff radius
        !   grid: the geometric shape of the grid 
        !   n1,n2,n3: counter on x,y,z
        ! output variables:
        !   ng: number of grids in the sphere
        !   gcart: cartesian coordinates of the grids
        !   gfrac: fractional coordinates of the grids
        !     
        real(8), intent(in) :: rcut, grid(3,3)
        integer, intent(in) :: ng
        integer, intent(in) :: nm1, nm2, nm3
        real(8), intent(out) :: gcart(3,ng)
        integer, intent(out) :: gfrac(3,ng)
        !  
        ! working variables:
        !   ig: counter on G vectors
        !   n: counter on k-grid point
        integer :: n1, n2, n3, ig
        !   g0: dummy G vector
        !   gg: |g0|
        real(8) :: g0(3),gg
        !   a*: Bravais vectors
        !   h*: reciprocal Bravais vectors
        real(8),dimension(3) :: g1, g2, g3
        !----------------------------------------------------------------
        g1 = grid(:,1); g2 = grid(:,2); g3 = grid(:,3)

        ! set grids such that gcart^2 < rcut, gcart(n1,n2,n3) =  n1*g1 + n2*g2 + n3*g3
        !
        ig = 0
        do n1 = -nm1, nm1
            do n2 = -nm2, nm2
                do n3 = -nm3, nm3
                    g0(:) = n1*g1(:) + n2*g2(:) + n3*g3(:)
                    gg = sqrt(g0(1)**2 + g0(2)**2 + g0(3)**2)
                    if ( gg <= rcut ) then
                        ig = ig + 1
                        gcart (:,ig) = g0(:)
                        gfrac(1,ig) = n1; gfrac(2,ig) = n2; gfrac(3,ig) = n3
                    end if
                end do
            end do
        end do
        
        if ( ig /= ng ) then
            print *, 'Mismatch in number of G-grid!',ig,ng
            stop
        end if
    end subroutine set_grid

    subroutine count_grid(rcut, grid, ng, nm1, nm2, nm3)
        ! input variables:
        !   rcut: cutoff radius
        !   grid: the geometric shape of the grid         
        real(8), intent(in) :: rcut, grid(3,3)
        integer, intent(out) :: ng, nm1, nm2, nm3
        ! working variables:
        !   ig: counter on G vectors
        !   n: counter on k-grid point
        integer :: n1, n2, n3
        real(8) :: vol
        real(8),dimension(3) :: h1, h2, h3  
        !   g0: dummy G vector
        !   gg: |g0|
        real(8) :: g0(3),gg       
        real(8),dimension(3) :: g1, g2, g3   

        g1 = grid(:,1); g2 = grid(:,2); g3 = grid(:,3)
        vol = dot_product(g1, cross(g2, g3))
        h1 = cross(g2, g3) / vol
        h2 = cross(g3, g1) / vol
        h3 = cross(g1, g2) / vol        
           
        ! estimate max fractional coordinates n1,n2,n3:
        ! n1 = rcut*h1 -> abs(n1) <= |G||h1|...
        !
        nm1 = nint (rcut*sqrt(h1(1)**2+h1(2)**2+h1(3)**2) + 0.5 )
        nm2 = nint (rcut*sqrt(h2(1)**2+h2(2)**2+h2(3)**2) + 0.5 )
        nm3 = nint (rcut*sqrt(h3(1)**2+h3(2)**2+h3(3)**2) + 0.5 )        
        !
        ! count number of grids
        !
        ng = 0
        do n1 = -nm1, nm1
            do n2 = -nm2, nm2
                do n3 = -nm3, nm3
                    g0(:) = n1*g1(:) + n2*g2(:) + n3*g3(:)
                    gg = sqrt(g0(1)**2 + g0(2)**2 + g0(3)**2)
                    if ( gg <= rcut ) ng = ng + 1
                end do
            end do
        end do
    end subroutine count_grid

    pure function cross(a, b) result(c)
        ! Compute cross product of two 3D vectors
        real(8), intent(in) :: a(3), b(3)
        real(8) :: c(3)
        c(1) = a(2)*b(3) - a(3)*b(2)
        c(2) = a(3)*b(1) - a(1)*b(3)
        c(3) = a(1)*b(2) - a(2)*b(1)
    end function cross    
end module utils