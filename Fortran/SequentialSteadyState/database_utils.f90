module database_utils
use globals
implicit none

    interface linear
        module procedure linear1        
        module procedure linear2        
    end interface linear

    interface bilinear
        module procedure bilinear2        
        module procedure bilinear4        
    end interface bilinear

    interface ix_fix
        module procedure ix_fix_from
        module procedure ix_fix_global
    end interface ix_fix

contains

!   function proportion(x,x0 
!              c1_min = real(ix_min) + (x1-xtb(ix_min))/(xtb(ix_min+1)-xtb(ix_min))

    function ix_fix_from(xtb,x1,c0,s1,default) result (c1)
    ! start from position ix, looking for table position both ways, stop at the nearest index 
    ! searching for index ip and fraction fip going with
    ! value x1
        integer                    :: s1
        real(kind=hp)              :: c1
        real(kind=hp), intent(in)  :: xtb(s1:)
        real(kind=hp), intent(in)  :: x1
        real(kind=hp), intent(in)  :: c0
        real(kind=hp), intent(in), optional  :: default 

        integer :: ix, iplus, imin, ix_min, ix_plus
        real(kind=hp) :: c1_min, c1_plus, x0, tol
        logical :: ja_min, ja_plus, do_min, do_plus
        integer :: imaximum, iminimum

        tol = 0.01*eps   ! todo: obtain a reasonable tolerance
        tol = 0.0001*eps ! todo: obtain a reasonable tolerance
        tol = 0.d0       ! zero tolerance ! 

        ! do not change index when change in x1 within tolerance 
        x0=linear1(xtb,c0,s1) 
        if (abs(x1-x0)<tol) then
           c1 = c0
           return
        endif 

        iminimum = lbound(xtb,dim=1)
        imaximum = ubound(xtb,dim=1)
        ix = floor(c0)

        if ((x1-xtb(ubound(xtb,1)))*(x1-xtb(lbound(xtb,1)))>=0.) then
            c1 = c0
        elseif ((x1-xtb(ix+1))*(x1-xtb(ix))<=0.) then
            if (abs(xtb(ix+1)-xtb(ix))>eps) then
                c1 = real(ix) + (x1-xtb(ix))/(xtb(ix+1)-xtb(ix))
            else
                c1 = c0
            endif
        else
            do_min = .True.
            do_plus= .True.
            ja_min = .False.
            ja_plus= .False.
            iplus = ix
            imin = ix
            do while( &
               (do_min.or.do_plus) .and.         &      ! still searching in at least one direction
               (.not.(ja_min.or.ja_plus)))              ! not yet found an interval in any direction
               if (do_plus) then
                  iplus = iplus + 1 
                  ja_plus = ((x1-xtb(iplus-1))*(x1-xtb(iplus))<=0)
                  do_plus = (iplus<imaximum)
               endif
               if (do_min) then
                  imin = imin - 1 
                  ja_min = ((x1-xtb(imin+1))*(x1-xtb(imin))<=0)
                  do_min = (imin>iminimum)
               endif
            enddo
            if (ja_plus .and. .not.ja_min) then ! nearest matching interval in positive direction
               ix = iplus-1
               c1 = real(ix) + (x1-xtb(ix))/(xtb(ix+1)-xtb(ix))
            elseif (ja_min .and. .not.ja_plus) then ! nearest matching interval in negative direction
               ix = imin
               c1 = real(ix) + (x1-xtb(ix))/(xtb(ix+1)-xtb(ix))
            elseif (ja_min .and. ja_plus) then ! positive and negative direction both successful
               ! regard the fractions, find which one is nearest 
               ix_min = imin
               ix_plus = iplus-1
               c1_min = real(ix_min) + (x1-xtb(ix_min))/(xtb(ix_min+1)-xtb(ix_min))
               c1_plus = real(ix_plus) + (x1-xtb(ix_plus))/(xtb(ix_plus+1)-xtb(ix_plus))
               c1 = merge (c1_min,c1_plus,(c1_plus-c0)>(c0-c1_min))
            else ! neither direction successful
               ! keep the default  
               c1 = default 
            endif
        endif
    end function ix_fix_from


    function ix_fix_global(xtb,x1,s1,default) result (c1)
    ! looking for table position,
    ! searching for index ip and fraction fip going with
    ! value x1
        real(kind=hp)              :: c1
        integer,       intent(in)  :: s1
        real(kind=hp), intent(in)  :: xtb(s1:)
        real(kind=hp), intent(in)  :: x1
        real(kind=hp), intent(in), optional  :: default 
        integer       :: ix, i1, i2
        real(kind=hp) :: fix, sgn

        i1 = lbound(xtb,dim=1) ! lower bound of index
        i2 = ubound(xtb,dim=1) ! upper bound of index
        if (abs(xtb(i2)-xtb(i1))<eps) then
            if (present(default)) then
                c1 = default
            else
                c1 = (i1 + i1)/2.d0
            endif
            return 
        endif
        sgn=sign(1.,xtb(i2)-xtb(i1))     ! -1. for descending, 1. for ascending x
        if ((x1-xtb(i2))*sgn>0.d0) then  ! beyond the table upper bound, 
           ix = i2-1                     ! take the one-but-highest index
           fix = 1.d0                    ! set weight to 1
        elseif ((x1-xtb(i1))*sgn<0.d0) then 
           ix = i1                       ! take the lowest index
           fix = 0.d0                    ! set weight to 0
        else
           do while(i2-i1>1)   ! narrow the interval, bisection-wise
              ix=floor((i1+i2)/2.)
              if ((xtb(ix)-x1)*(xtb(i2)-x1)>0) then
                 i2=ix
              else
                 i1=ix
              endif
           enddo
           ix = min(i1,i2)     ! take the midpoint of the resulting interval as final index
           if (abs(xtb(ix+1)-xtb(ix))<eps) then      ! protection against zero division
              fix = 0.d0
           else 
              fix=(x1-xtb(ix))/(xtb(ix+1)-xtb(ix))   ! use the obtained index, derive the fraction
           endif
        endif
        c1 = ix + fix                          ! for convenience combine index and fraction in a single float8, (still sufficiently accurate)
    end function ix_fix_global


    function linear1(tb,cx,sx) result (z)
        real(kind=hp)              :: z
        real(kind=hp), intent(in)  :: tb(:)
        real(kind=hp), intent(in)  :: cx
        integer,       intent(in)  :: sx
        integer :: ix
        real(kind=hp) :: fx
        ix = floor(cx)
        fx = cx - ix
        z = linear2(tb,ix,fx,sx)
    end function linear1

    function linear2(tb,ix,fx,sx) result (z)
        real(kind=hp)              :: z
        integer,       intent(in)  :: sx
        real(kind=hp), intent(in)  :: tb(sx:)
        integer,       intent(in)  :: ix
        real(kind=hp), intent(in)  :: fx
        z = tb(ix+1)*fx + tb(ix)*(1.-fx) ! interpolated value
    end function linear2

    function bilinear2(tb,cx,cy,sx,sy) result (z)
        real(kind=hp)              :: z
        real(kind=hp), intent(in)  :: tb(:,:)
        real(kind=hp), intent(in)  :: cx
        real(kind=hp), intent(in)  :: cy
        integer,       intent(in)  :: sx
        integer,       intent(in)  :: sy
        integer :: ix, iy
        real(kind=hp) :: fx, fy
        ix = floor(cx) 
        iy = floor(cy)
        fx = cx - ix
        fy = cy - iy
        z = bilinear4(tb,ix,iy,fx,fy,sx,sy)
    end function bilinear2

    function bilinear4(tb,ix,iy,fx,fy,sx,sy) result (z)
        real(kind=hp)              :: z
        integer,       intent(in)  :: ix, iy
        real(kind=hp), intent(in)  :: fx, fy
        integer,       intent(in)  :: sx, sy
        real(kind=hp), intent(in)  :: tb(sx:,sy:)
        z = (1.-fx)*(1.-fy)*tb(ix  ,iy  ) &
          +     fx *(1.-fy)*tb(ix+1,iy  ) &
          + (1.-fx)*    fy *tb(ix  ,iy+1) &
          +     fx *    fy *tb(ix+1,iy+1)
    end function bilinear4

end module database_utils

