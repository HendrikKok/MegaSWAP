module unsaturated_zone
use globals
use database_io
use database_utils
implicit none

    type :: t_unsa
        integer                    :: nbox, nnod    ! number of boxes, number of nodes
        real(kind=hp), allocatable :: qmv(:)
        real(kind=hp), allocatable :: sv(:), sv0(:)
        real(kind=hp), pointer     :: phead(:) => null()
        real(kind=hp), allocatable :: phi(:)
        real(kind=hp), pointer     :: dt => null()
        real(kind=hp)              :: gam
        real(kind=hp)              :: top
        integer                    :: sl
        integer                    :: maxbox
        logical, allocatable       :: submerged(:)
        real(kind=hp), allocatable :: storage(:)
        type(t_database), pointer  :: unsa_db => null()
    contains
        procedure, pass :: initialize        => t_unsa_init                ! init unsaturated zone instance
        procedure, pass :: update            => t_unsa_update              ! update unsaturated soil
        procedure, pass :: finalize_timestep => t_unsa_finalize_timestep
        procedure, pass :: set_db            => t_unsa_set_db              ! sets database
        procedure, pass :: nonsubmerged      => t_unsa_nonsubmerged        ! returns the maximum non-submerged box
        procedure, pass :: gwl2gamma         => t_unsa_gwl2gamma           ! given a groundwater level, return groundwater level index 
        procedure, pass :: gamma2gwl         => t_unsa_gamma2gwl           ! given a groundwater level index, return groundwater level
        procedure, pass :: gamma2storage     => t_unsa_gamma2storage       ! given a groundwater level index, return total storage in column
        procedure, pass :: get_sc1           => t_unsa_get_sc1             ! given a groundwater level index, estimage storage coefficient
        procedure, pass :: get_theta         => t_unsa_get_moisture_content! convert the sv-values to non-dimensional moisture content
    end type t_unsa
    
contains

    function t_unsa_gwl2gamma(uz, gwl) result (gamma)
        real(kind=hp)                 :: gamma
        class(t_unsa), intent(inout)  :: uz
        real(kind=hp), intent(in)     :: gwl             ! groundwater level
        gamma = uz%unsa_db%dpgw2gamma(uz%top-gwl)
    end function t_unsa_gwl2gamma

    function t_unsa_gamma2gwl(uz, gamma) result (gwl)
        real(kind=hp)                 :: gwl
        class(t_unsa), intent(inout)  :: uz
        real(kind=hp), intent(in)     :: gamma           ! groundwater index
        gwl = uz%top-uz%unsa_db%gamma2dpgw(gamma)
    end function t_unsa_gamma2gwl

    function t_unsa_init(uz, phead, gwl, dbptr) result (success)
        logical                       :: success
        class(t_unsa), intent(inout)  :: uz
        real(kind=hp), intent(in)     :: phead           ! initial head
        real(kind=hp), intent(in)     :: gwl             ! initial groundwater level
        type(t_database), pointer     :: dbptr           ! Database pointer
        integer :: ibox, lgam, lphi

        success = .False.
        uz%unsa_db => dbptr

        uz%nbox = dbptr%nbox
        if (.not.realloc(uz%qmv,1,uz%nbox)) return
        if (.not.realloc(uz%sv,1,uz%nbox)) return
        if (.not.realloc(uz%sv0,1,uz%nbox)) return
        if (.not.realloc(uz%phi,1,uz%nbox)) return
        if (.not.realloc(uz%storage,1,uz%nbox)) return
        if (.not.realloc(uz%submerged,1,uz%nbox)) return
        uz%storage = 0.
        uz%phead = phead
        uz%submerged = .False.

        ! get the groundwater level index gam for gwl
        uz%gam = uz%unsa_db%dpgw2gamma(uz%top-gwl)

        lgam = lbound(uz%unsa_db%svtb, dim=2)  
        lphi = lbound(uz%unsa_db%svtb, dim=3)

        do ibox=1,uz%nbox
            ! get the pressure indices phi(:) for phead
            uz%phi(ibox) = phead2ndx(uz%phead(ibox),uz%unsa_db%ddpptb)
            ! set storage values sv(:) for indices set gam, phi(:)
            uz%sv(ibox) = bilinear(uz%unsa_db%svtb(ibox,:,:),uz%gam,uz%phi(ibox),lgam,lphi)
            ! preserve old storage values sv(:)
            uz%sv0(:) = uz%sv(:)    ! check whether this can be removed 
            ! set fluxes qmv(:)
            uz%qmv(ibox) = bilinear(uz%unsa_db%qmrtb(:,:),uz%gam,uz%phi(ibox),lgam,lphi)
        enddo
        success = .True.
    end function t_unsa_init

    function t_unsa_update(uz, qrch, gwl) result (dsv)
        real(kind=hp)                           :: dsv              ! increase in column total storage 
        class(t_unsa), intent(inout)            :: uz
        real(kind=hp), intent(in)               :: qrch             ! recharge flux topmost box
        real(kind=hp), intent(in)               :: gwl              ! groundwater level

        real(kind=hp) :: sigma, qin, gam_local
        integer :: ibox, lgam, lphi

        uz%maxbox = uz%nonsubmerged(gwl) 
        lgam = lbound(uz%unsa_db%svtb, dim=2)  
        lphi = lbound(uz%unsa_db%svtb, dim=3)

        do ibox=1, uz%maxbox
            uz%phi(ibox) = phead2ndx(uz%phead(ibox),uz%unsa_db%ddpptb)
            gam_local = uz%gam
            ! tbd: ig_local, gw level with largest qmr flux, adjust local
            if (ibox==1) then
                qin = -qrch
            else
                qin = uz%qmv(ibox-1)
            endif
            sigma = uz%sv0(ibox) - qin * uz%dt
            call uz%unsa_db%sigma2phi(sigma, uz%phi(ibox), uz%gam, uz%dt, ibox, default=uz%phi(ibox))
            if (uz%phi(ibox)<0) then
                uz%phead(ibox) = linear(uz%unsa_db%ptb,uz%phi(ibox),lphi)
            else
                uz%phead(ibox) = -(10**(uz%phi(ibox)*uz%unsa_db%ddpptb))/m2cm
            endif
            uz%sv(ibox) = bilinear(uz%unsa_db%svtb(ibox,:,:),gam_local,uz%phi(ibox),lgam,lphi)
            if (ibox==1) then
               uz%qmv(ibox) = (uz%sv(ibox) - uz%sv0(ibox))/uz%dt - qrch
            else
               uz%qmv(ibox) = (uz%sv(ibox) - uz%sv0(ibox))/uz%dt + uz%qmv(ibox-1) 
            endif
        enddo
        dsv = (sum(uz%sv(1:uz%maxbox))-sum(uz%sv0(1:uz%maxbox)))

        ! default head, gwl dependend, for submerged boxes 
        uz%phead(uz%maxbox+1:) = (gwl - uz%top) + 0.5 * uz%unsa_db%dprzsl
        !uz%phead(:) = (gwl-uz%top) + 0.5 * uz%unsa_db%dprzsl   ! default pressure head submerged boxes
    end function t_unsa_update

    function t_unsa_gamma2storage(uz, gam) result (stotal)
        real(kind=hp)  :: stotal 
        class(t_unsa), intent(inout) :: uz
        real(kind=hp), intent(in)    :: gam 
        real(kind=hp)  :: sv, gwl  
        integer :: ibox, lgam, lphi
        stotal = 0.d0
        gwl = uz%gamma2gwl(gam)
        uz%maxbox = uz%nonsubmerged(gwl) ! gwl calculated from gam
        lgam = lbound(uz%unsa_db%svtb, dim=2)  
        lphi = lbound(uz%unsa_db%svtb, dim=3)
        do ibox=1,uz%maxbox
           sv = bilinear(uz%unsa_db%svtb(ibox,:,:),gam,uz%phi(ibox),lgam,lphi)
           stotal = stotal + sv  
        enddo
    end function t_unsa_gamma2storage

    function t_unsa_get_sc1(uz, gamma) result (sc1)
        real(kind=hp)  :: sc1 
        class(t_unsa), intent(inout) :: uz
        real(kind=hp) ::sv0, sv1, gwl0, gwl1, gamma, dgamma
        real(kind=hp), parameter :: sc1_min = 1.e-04
        real(kind=hp) :: gamma0, gamma1,lgam, ugam
        logical, parameter :: newinterp = .True.
        dgamma = 0.5d0

        if (newinterp) then
           lgam = real(lbound(uz%unsa_db%dpgwtb,dim=1))
           ugam = real(ubound(uz%unsa_db%dpgwtb,dim=1))
           gamma0 = max(gamma-dgamma,real(lgam))
           gamma1 = min(gamma+dgamma,real(ugam)-eps)
           sv0 = uz%gamma2storage(gamma0)
           sv1 = uz%gamma2storage(gamma1)   ! central difference
           gwl0 = uz%gamma2gwl(gamma0)
           gwl1 = uz%gamma2gwl(gamma1)
        else
           gamma0 = floor(gamma)
           gamma1 = gamma0 + 1.d0
           sv0 = uz%gamma2storage(gamma0)
           sv1 = uz%gamma2storage(gamma1)
           gwl0 = uz%gamma2gwl(gamma0)
           gwl1 = uz%gamma2gwl(gamma1)
        endif

        if (abs(gwl1-gwl0)>eps) then
            sc1 = (sv1 - sv0)/(gwl1 - gwl0)
            sc1 = max(min(sc1, 1.d0),sc1_min)
        else
            sc1 = sc1_min
        endif
    end function t_unsa_get_sc1


    subroutine t_unsa_set_db(uz, dbptr)
        class(t_unsa), intent(inout) :: uz
        type (t_database), pointer   :: dbptr
        uz%unsa_db => dbptr
        uz%nbox = max(uz%nbox, ubound(dbptr%svtb, dim=1))   ! limited to the number of boxes in the svtb table
    end subroutine t_unsa_set_db

    subroutine t_unsa_finalize_timestep(uz,gwl,qmodf)
        class(t_unsa), intent(inout) :: uz
        real(kind=hp), intent(in)    :: gwl              ! groundwater level
        real(kind=hp), intent(in)    :: qmodf
        integer :: ibox, lgam, lphi, nbox, lastbox
        nbox = uz%nbox

        lgam = lbound(uz%unsa_db%svtb, dim=2)  
        lphi = lbound(uz%unsa_db%svtb, dim=3)
        uz%gam = uz%unsa_db%dpgw2gamma(uz%top-gwl)
        ! update storage deficit
        do ibox=1, uz%nbox
            uz%sv(ibox) = bilinear(uz%unsa_db%svtb(ibox,:,:),uz%gam,uz%phi(ibox),lgam,lphi)
        enddo

        ! update storage fluxes qmv
        uz%qmv(:) = 0.d0
        uz%qmv(nbox) = qmodf
        uz%qmv(nbox-1) = -(uz%sv(nbox) - uz%sv0(nbox)) / uz%dt + qmodf
        do ibox=nbox-2,1,-1
            uz%qmv(ibox) = -(uz%sv(ibox+1) - uz%sv0(ibox+1)) / uz%dt + uz%qmv(ibox+1)
        enddo

        ! update pressure heads prz 
        lastbox = uz%nonsubmerged(gwl)                     ! last non-submerged box
        uz%submerged(:lastbox) = .False.    ! These boxes are not submerged
        uz%submerged(lastbox+1:) = .True.   ! These boxes are submerged
        do ibox=1, lastbox
            call uz%unsa_db%sv2phi(uz%sv(ibox), uz%phi(ibox), uz%gam, ibox, default=uz%phi(ibox))
            if (uz%phi(ibox)<0) then
                uz%phead(ibox) = linear(uz%unsa_db%ptb,uz%phi(ibox),lphi)
            else
                uz%phead(ibox) = -(10**(uz%phi(ibox)*uz%unsa_db%ddpptb))/m2cm
            endif
        enddo
        ! for submerged boxes automatically phead becomes equal to the deepest non-submerged box
        ! uz%phead(lastbox+1:nbox) = uz%phead(lastbox) ! copy the value of the deepest non-submerged box to submerged boxes.
        uz%sv0(:) = uz%sv(:)
    end subroutine t_unsa_finalize_timestep

    function t_unsa_nonsubmerged(uz,gwl) result (maxbox)
        integer                      :: maxbox
        real(kind=hp), intent(in)    :: gwl              ! groundwater level
        class(t_unsa), intent(inout) :: uz
        integer :: ibox
        do ibox=1, uz%nbox
            maxbox=ibox
            if (gwl>uz%unsa_db%hbotb(ibox)) then
                exit
            endif   ! this was the deepest non-submerged box
        enddo
    end function t_unsa_nonsubmerged

    subroutine t_unsa_get_moisture_content(uz, theta_box)
        class(t_unsa), intent(inout) :: uz
        real(kind=hp), intent(out)   :: theta_box(:)
        integer :: ibox, lgam, lphi
        lgam = lbound(uz%unsa_db%thetatb,dim=2)
        lphi = lbound(uz%unsa_db%thetatb,dim=3)
        do ibox=1,min(size(theta_box),uz%nbox)
           theta_box(ibox) = bilinear(uz%unsa_db%thetatb(ibox,:,:),uz%gam,uz%phi(ibox),lgam,lphi)
        enddo
    end subroutine t_unsa_get_moisture_content
end module unsaturated_zone
