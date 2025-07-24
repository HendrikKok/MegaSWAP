module database_io
use globals
use database_utils
use netcdf
implicit none

integer, parameter :: NMAXSPU = 370
integer, parameter :: NMAXRZ = 31

    ! database-set containing databases for a matrix of soiltype and rootzone depths
    ! database for a single soiltype and rootzone depth
    type :: t_databaseSet
        character(len=:),allocatable             :: unsa_path
        real(kind=hp),    allocatable            :: rzdepth(:)
        type(t_databaseptr), allocatable         :: dbs(:,:)             ! 2D-array of pointers to database instances
    contains
        procedure, pass :: readNCset => t_databaseSet_readNCset          ! Read database contents from netCDF4 files
        procedure, pass :: getDbPtr => t_databaseSet_getDbPtr            ! Return pointer to database instance within database set
    end type t_databaseSet

    type :: t_database
        integer                       :: irz = 0 
        integer                       :: spu = 0 
        character(len=:), allocatable :: dbpath
        integer                       :: nbox  
        real(kind=hp)                 :: ddgwtb
        real(kind=hp)                 :: ddpptb
        real(kind=hp)                 :: dpczsl
        real(kind=hp)                 :: dprzsl
        integer         , allocatable :: igdc(:)
        real(kind=hp)   , allocatable :: hbotb(:)
        real(kind=hp)   , allocatable :: dpgwtb(:)
        real(kind=hp)   , allocatable :: qmrtb(:,:)
        real(kind=hp)   , allocatable :: svtb(:,:,:)
        real(kind=hp)   , allocatable :: thetatb(:,:,:)
        real(kind=hp)   , allocatable :: ptb(:)
    contains
        procedure, pass :: readNC => t_database_readNC          ! Read database contents from netCDF4 files
        procedure, pass :: sigma2phi => t_database_sigma2phi    ! Lookup phi index in svtb-qmrtb*dtgw given index gamma and sigma
        procedure, pass :: sv2phi => t_database_sv2phi          ! Lookup phi index in svtb given index gamma and sv
        procedure, pass :: dpgw2gamma => t_database_dpgw2gamma  ! groundwater depth to index gamma
        procedure, pass :: gamma2dpgw => t_database_gamma2dpgw  ! index gamma to groundwater depth
    end type t_database

    type :: t_databasePtr
        type (t_database), pointer :: ptr => null()
    end type t_databasePtr
    
contains

    function nf90chk(ierr, lun, note, ncid) result (failed)
    implicit none
    logical                      :: failed
    integer, intent(in)          :: ierr
    integer, intent(in)          :: lun
    integer, intent(in), optional:: ncid
    character(len=*), intent(in) :: note
    integer :: err
    if (ierr/=NF90_NOERR) then
       write(lun,*) trim(note)
       write(lun,*) trim(nf90_strerror(ierr))
       write(lun,*)
       failed = .True.
       if (present(ncid)) then
          err = nf90_close(ncid)
       endif
    else
       failed = .False.
    endif
    end function

    function t_databaseSet_readNCset(dbset, unsa_path, select_spu) result (success)
        logical :: success
        class(t_databaseSet), intent(inout) :: dbset
        character(*),         intent(in)    :: unsa_path
        logical, optional,    intent(in)    :: select_spu(:) ! logical array of selected soil types
    
        character(len=:), allocatable :: filname
        character(len=60) :: regel
        character(len=7) dum
        integer :: ncid, lun
        integer :: spu, rz, idrz
        integer :: ierr
        logical :: exists
        class(t_database), pointer :: db => null()
        real(kind=hp) :: drz
    
        success = .False.
        dbset%unsa_path = unsa_path

        allocate(dbset%dbs(NMAXRZ, NMAXSPU))
        allocate(dbset%rzdepth(NMAXRZ))

        ! numbered list of rootzone depths into array
        inquire(file=trim(dbset%unsa_path)//'/rootzones.txt',exist=exists)
        if (exists) then
           open(file=trim(dbset%unsa_path)//'/rootzones.txt',status='OLD',newunit=lun)
           ierr = 0
           rz=0
           do while(ierr==0)
              read(lun,'(a60)',iostat=ierr) regel
              if (ierr==0) then
                 read(regel,*) idrz, drz
                 dbset%rzdepth(rz+1) = drz
                 rz = rz + 1 
              endif
           enddo
           close(lun)
        else
           return ! unsuccesfull, missing rootzone depths table rootzones.txt
        endif

        ! read table contents from files
        do spu=1, NMAXSPU
           if (present(select_spu)) then
              if (.not.select_spu(spu)) then
                 cycle
              endif
           endif
           do rz=1, NMAXRZ
!             write(dum,'(i3.3,a,i3.3)') spu, '_', rz
              write(dum,'(i3.3,a,i3.3)') spu, '_', nint(dbset%rzdepth(rz)*100)
              filname=trim(dbset%unsa_path)//'/unsa_'//dum//'.nc'
              ierr = nf90_open(trim(filname), NF90_NOWRITE, ncid)
              if (ierr.eq.NF90_NOERR) then
                 write(0,*) 'reading '//filname//' ...' 
                 ! create database instance
                 allocate(db)
                 if (.not.db%readNC(ncid)) then
                     return           ! problem occurred reading this particular database
                 endif
                 db%dbpath = filname
                 ! ToDo:check if root zone depth of the file matches with the expected depth
                 if (abs(db%dprzsl-dbset%rzdepth(rz))>1e-3) then
                    ! non-matching rz 
                 endif
                 dbset%dbs(rz,spu)%ptr => db
                 db%irz = rz
                 db%spu = spu
              endif
              ierr = nf90_close(ncid) ! close open netcdf file
           enddo ! rootzones on nc-files
        enddo ! soiltypes on nc-files
        success = .True.
    end function t_databaseSet_readNCset

    function t_databaseSet_getDbPtr(dbset, sl_tgt, drz) result (dbptr)
        type(t_database),     pointer       :: dbptr ! pointer to database instance within the database set
        class(t_databaseSet), intent(inout) :: dbset
        integer,              intent(in)    :: sl_tgt! absolute soil type number
        real(kind=hp), intent(in)           :: drz   ! rootzone thickness
        integer                             :: rz_tgt, irz 
        real(kind=hp), parameter :: tol=1.e-05
        rz_tgt = 0
        do irz=1,size(dbset%rzdepth)
            if (abs(dbset%rzdepth(irz)-drz)<=tol) then
                rz_tgt = irz
                exit
            endif
        enddo
        if ((sl_tgt>0) .and. (rz_tgt>0)) then
            dbptr => dbset%dbs(rz_tgt, sl_tgt)%ptr
        else
            dbptr => null()
        endif
    end function t_databaseSet_getDbPtr

    function t_database_readNC(db,ncid) result (success)
        logical :: success
        class(t_database), intent(inout) :: db
        integer,           intent(in)    :: ncid   ! netcdf file handle
    
        integer :: dimid_ib, dimid_igdc
        integer :: varid_qmr, varid_sv, varid_dpgw, varid_igdc, varid_theta
        integer :: varid_hbotb
        integer :: sl, rz, igdcmn, igdcmx
        integer :: ierr
        integer :: nxb, nuip, nlip, nuig, nlig, nxlig, nxuig
    
        success = .False.
        ierr = nf90_inq_dimid(ncid,'ib',dimid_ib)
        ierr = nf90_inquire_dimension(ncid, dimid_ib, len=nxb)
        ierr = nf90_inq_dimid(ncid,'igdc',dimid_igdc)
        ierr = nf90_get_att(ncid, NF90_GLOBAL, 'igdcmn', igdcmn)
        ierr = nf90_get_att(ncid, NF90_GLOBAL, 'igdcmx', igdcmx)
        ierr = nf90_get_att(ncid, NF90_GLOBAL, 'ddpptb', db%ddpptb)
        ierr = nf90_get_att(ncid, NF90_GLOBAL, 'ddgwtb', db%ddgwtb)
        ierr = nf90_get_att(ncid, NF90_GLOBAL, 'dpczsl', db%dpczsl)
        ierr = nf90_get_att(ncid, NF90_GLOBAL, 'dprzsl', db%dprzsl)
        ierr = nf90_get_att(ncid, NF90_GLOBAL, 'nuip', nuip)
        ierr = nf90_get_att(ncid, NF90_GLOBAL, 'nlip', nlip)
        ierr = nf90_get_att(ncid, NF90_GLOBAL, 'nlig', nlig)
        ierr = nf90_get_att(ncid, NF90_GLOBAL, 'nuig', nuig)
        ierr = nf90_get_att(ncid, NF90_GLOBAL, 'nxlig', nxlig)
        ierr = nf90_get_att(ncid, NF90_GLOBAL, 'nxuig', nxuig)
        ierr = nf90_get_att(ncid, NF90_GLOBAL, 'sl', sl)    ! soil type number
        ierr = nf90_get_att(ncid, NF90_GLOBAL, 'rz', rz)    ! rootzone depth number
    
        ! allocate arrays in db
        success = realloc(db%qmrtb, nxlig-1,    nuig,    nlip, nuip)
        success = realloc(db%svtb,        1,     nxb, nxlig-1, nuig, nlip, nuip)
        success = realloc(db%thetatb,     1,     nxb, nxlig-1, nuig, nlip, nuip)   ! theta at box resolution
        success = realloc(db%hbotb,       0,     nxb)
        success = realloc(db%igdc,  igdcmn,   igdcmx)
        success = realloc(db%dpgwtb,nxlig-1,    nuig)   

        ! initialize contents of all tables (for missing soiltypes)
        db%svtb = DMISS
        db%hbotb = DMISS
        db%qmrtb = DMISS
        db%dpgwtb = DMISS
        db%igdc = DMISS
        db%nbox = nxb
    
        ! inquire variable id
        ierr = nf90_inq_varid(ncid, 'svtb', varid_sv)
        ierr = nf90_inq_varid(ncid, 'thetatb', varid_theta)
        ierr = nf90_inq_varid(ncid, 'hbotb', varid_hbotb)
        ierr = nf90_inq_varid(ncid, 'igdc', varid_igdc)
        ierr = nf90_inq_varid(ncid, 'dpgwtb', varid_dpgw)
        ierr = nf90_inq_varid(ncid, 'qmrtb', varid_qmr)

        ! read variable data
        ierr = nf90_get_var(ncid, varid_sv, db%svtb(:,:,:))
        ierr = nf90_get_var(ncid, varid_theta, db%thetatb(:,:,:))
        ierr = nf90_get_var(ncid, varid_hbotb, db%hbotb(1:))
        ierr = nf90_get_var(ncid, varid_igdc, db%igdc(:))
        ierr = nf90_get_var(ncid, varid_dpgw, db%dpgwtb(:))
        ierr = nf90_get_var(ncid, varid_qmr, db%qmrtb(:,:))

        db%hbotb(0) = 0.d0
        db%hbotb(1) = db%hbotb(0) - db%dprzsl
        db%hbotb(2) = db%hbotb(1) - db%dpczsl
        ierr = make_ptb_values(db) 
        success = .True.
    end function t_database_readNC

    function make_ptb_values(db) result (success)
        logical :: success
        class (t_database), intent(inout) :: db
        integer :: ip, nlip, nuip
        nlip = lbound(db%qmrtb,dim=2)
        nuip = ubound(db%qmrtb,dim=2)
    
        success = .False.
        if (allocated(db%ptb)) deallocate(db%ptb)
        allocate(db%ptb(nlip:nuip))
        do ip=nlip,0
           db%ptb(ip)=-ip*db%ddpptb
        enddo
        do ip=1,nuip
           db%ptb(ip)=-10.**(ip*db%ddpptb)/m2cm
        enddo
    end function make_ptb_values

    subroutine t_database_sv2phi(db, sv, phi, gamma, ibox, default)
        class (t_database),intent(inout) :: db     ! database struct     
        real(hp),          intent(inout) :: phi    ! pressure index
        real(hp),          intent(in)    :: gamma  ! groundwater index
        real(hp),          intent(in)    :: sv     ! the sv value to be found 
        integer,           intent(in)    :: ibox   ! box number
        real(hp),          intent(in), optional    :: default  ! default index gamma
        real(hp), allocatable :: sv1d(:)
        real(hp) :: fig 
        integer  :: ig
        save sv1d

        ig = floor(gamma)
        fig = gamma - ig
        if (.not.allocated(sv1d)) then
            allocate(sv1d(lbound(db%qmrtb,dim=2):ubound(db%qmrtb,dim=2)))
        endif
        sv1d = (db%svtb(ibox,ig,:) * (1.-fig) + db%svtb(ibox,ig+1,:) * fig) 
        phi = ix_fix(sv1d,sv,phi,lbound(sv1d,dim=1),default=default)
!       phi = ix_fix(sv1d,sv,lbound(sv1d,dim=1),default=default)
    end subroutine t_database_sv2phi

    subroutine t_database_sigma2phi(db, sigma, phi, gamma, dtgw, ibox, default)
        class (t_database),intent(inout) :: db     ! database struct     
        real(hp),          intent(inout) :: phi    ! pressure index
        real(hp),          intent(in)    :: gamma  ! groundwater index
        real(hp),          intent(in)    :: sigma  ! the sigma value to be found 
        real(hp),          intent(in)    :: dtgw   ! groundwater time step
        integer,           intent(in)    :: ibox   ! box number
        real(hp),          intent(in), optional    :: default  ! default index gamma
        real(hp), allocatable :: sigma1d(:)
        integer  :: ig
        real(hp) :: fig
        save sigma1d

        ig = floor(gamma)
        fig = gamma - ig
        if (.not.allocated(sigma1d)) then
            allocate(sigma1d(lbound(db%qmrtb,dim=2):ubound(db%qmrtb,dim=2)))
        endif
        sigma1d = (db%svtb(ibox,ig,:)   - (db%qmrtb(ig,:)))   * (1.-fig) &
                + (db%svtb(ibox,ig+1,:) - (db%qmrtb(ig+1,:))) * fig
        phi = ix_fix(sigma1d,sigma,phi,lbound(sigma1d,dim=1),default=default)
!       phi = ix_fix(sigma1d,sigma,lbound(sigma1d,dim=1),default=default)
    end subroutine t_database_sigma2phi

    function t_database_dpgw2gamma(db, dpgw) result (gamma)
        real(hp) :: gamma
        class (t_database),intent(in) :: db     ! database struct     
        real(hp),          intent(in) :: dpgw   ! groundwater depth relative to surface
        integer  :: igk, index
        real(hp) :: figk
        logical, parameter :: alternative = .True.
        if (alternative) then
           gamma = ix_fix(db%dpgwtb,dpgw,-2,gamma)   !! experimental
        else
           index = min(ubound(db%igdc, dim=1) - 1,         &
                       max(lbound(db%igdc, dim=1),         &
                          floor((dpgw + eps) / db%ddgwtb)))
           igk = db%igdc(index)
           figk = (dpgw - db%dpgwtb(igk)) / (db%dpgwtb(igk+1) - db%dpgwtb(igk))
           figk = min(1.d0, max(0.d0, figk))
           gamma = figk + igk
        endif
    end function t_database_dpgw2gamma

    function t_database_gamma2dpgw(db, gamma) result (dpgw)
        real(hp) :: dpgw                        ! groundwater depth relative ro surface
        class (t_database),intent(in) :: db     ! database struct     
        real(hp),          intent(in) :: gamma  ! groundwater level index
        integer  :: igk
        real(hp) :: figk

        igk = floor(gamma)
        figk = gamma - igk
        dpgw = db%dpgwtb(igk) + figk * (db%dpgwtb(igk+1) - db%dpgwtb(igk))
    end function t_database_gamma2dpgw

    function phead2ndx(phead, ddpptb) result (phi)
        real(kind=hp) :: phi
        real(kind=hp), intent(in) :: phead 
        real(kind=hp), intent(in) :: ddpptb
        if (phead>0.) then
            phi = -phead / ddpptb
        elseif (phead>=-0.01) then
            phi = 0.
        else
            phi = log10(-100. * phead) / ddpptb
        endif
    end function phead2ndx

    function ndx2phead(phi, ddpptb) result (phead)
        real(kind=hp) :: phead
        real(kind=hp), intent(in) :: phi
        real(kind=hp), intent(in) :: ddpptb
        if (phi<0.) then
            phead = -phi * ddpptb
        else
            phead = -0.01 * (10.**(phi * ddpptb))
        endif
    end function ndx2phead

end module database_io
