module globals
implicit none
        integer, parameter :: hp=8

        real(kind=hp), parameter :: DMISS  = -999.        
        real(kind=hp)            :: m2cm = 100.
        real(kind=hp)            :: eps = 0.1e-6

    interface realloc
        module procedure realloc1d_real
        module procedure realloc2d_real
        module procedure realloc3d_real
        module procedure realloc4d_real
        module procedure realloc1d_int
        module procedure realloc1d_logical
    end interface realloc

contains

    function realloc1d_real(arr,lbnd,ubnd) result (success)
        logical :: success
        real(kind=hp), intent(inout), allocatable :: arr(:)
        integer, intent(in) :: lbnd
        integer, intent(in) :: ubnd
        success = .False.
        if (allocated(arr)) deallocate(arr)
        allocate(arr(lbnd:ubnd))
        success = .True.
    end function realloc1d_real

    function realloc2d_real(arr,lbnd1,ubnd1,lbnd2,ubnd2) result (success)
        logical :: success
        real(kind=hp), intent(inout), allocatable :: arr(:,:)
        integer, intent(in) :: lbnd1, lbnd2
        integer, intent(in) :: ubnd1, ubnd2
        success = .False.
        if (allocated(arr)) deallocate(arr)
        allocate(arr(lbnd1:ubnd1,lbnd2:ubnd2))
        success = .True.
    end function realloc2d_real

    function realloc3d_real(arr,lbnd1,ubnd1,lbnd2,ubnd2,lbnd3,ubnd3) result (success)
        logical :: success
        real(kind=hp), intent(inout), allocatable :: arr(:,:,:)
        integer, intent(in) :: lbnd1, lbnd2, lbnd3
        integer, intent(in) :: ubnd1, ubnd2, ubnd3
        success = .False.
        if (allocated(arr)) deallocate(arr)
        allocate(arr(lbnd1:ubnd1,lbnd2:ubnd2,lbnd3:ubnd3))
        success = .True.
    end function realloc3d_real

    function realloc4d_real(arr,lbnd1,ubnd1,lbnd2,ubnd2,lbnd3,ubnd3,lbnd4,ubnd4) result (success)
        logical :: success
        real(kind=hp), intent(inout), allocatable :: arr(:,:,:,:)
        integer, intent(in) :: lbnd1, lbnd2, lbnd3, lbnd4
        integer, intent(in) :: ubnd1, ubnd2, ubnd3, ubnd4
        success = .False.
        if (allocated(arr)) deallocate(arr)
        allocate(arr(lbnd1:ubnd1,lbnd2:ubnd2,lbnd3:ubnd3,lbnd4:ubnd4))
        success = .True.
    end function realloc4d_real

    function realloc1d_int(arr,lbnd,ubnd) result (success)
        logical :: success
        integer, intent(inout), allocatable :: arr(:)
        integer, intent(in) :: lbnd
        integer, intent(in) :: ubnd
        success = .False.
        if (allocated(arr)) deallocate(arr)
        allocate(arr(lbnd:ubnd))
        success = .True.
    end function realloc1d_int

    function realloc1d_logical(arr,lbnd,ubnd) result (success)
        logical :: success
        logical, intent(inout), allocatable :: arr(:)
        integer, intent(in) :: lbnd
        integer, intent(in) :: ubnd
        success = .False.
        if (allocated(arr)) deallocate(arr)
        allocate(arr(lbnd:ubnd))
        success = .True.
    end function realloc1d_logical

end module globals
