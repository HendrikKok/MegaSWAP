!  ExampleSSS.f90 
!
!  FUNCTIONS:
!  ExampleSSS - Instructive application of sequential steady state class
!

!****************************************************************************
!
!  PROGRAM: ExampleSSS
!
!  PURPOSE:  Illustrates the use of the sequential steady state class simulating the unsaturated zone
!
!****************************************************************************

program ExampleSSS_coupled

use globals
use sss
use database_io
use dumpres
implicit none

type (t_SequentialSteadyStateSingle),  dimension(:), allocatable  :: svat
type (t_sssparam),                     dimension(:), allocatable  :: param
type(t_databaseSet),                                target        :: dbset

! results to netcdf file for check
type (t_dumpnc) :: dump
type (t_results), allocatable :: results(:)
character(len=50) :: unsa_dbname
integer :: spu
integer :: irz

character(len=:),                      allocatable                :: dbpath
real(kind=hp),    dimension(:,:),      allocatable                :: theta, h
real(kind=hp),    dimension(:),        allocatable                :: pond, qrunoff, qbot, qmodf, sc1
real(kind=hp),    dimension(:,:),      allocatable                :: reva
real(kind=hp)                                                     :: gwl, dtgw
logical                                                           :: suc6
integer                                                           :: n_svats, itime, iter, k, k1, k2, iun1, iun2, i
integer,          parameter                                       :: ntime = 200
integer,          parameter                                       :: niter = 6
integer,          parameter                                       :: nbox = 18
integer,          parameter                                       :: nnod = 18

real(kind=hp),    dimension(ntime)                                :: qrain, qevap, qrot, times, gwls
logical,          dimension(:),        allocatable                :: select_spu
logical,          dimension(:,:),        allocatable              :: submerged
logical                                                           :: external_input

! allocate
n_svats = 2
allocate(param(n_svats))
allocate(svat(n_svats))
allocate(pond(n_svats))
allocate(qrunoff(n_svats))
allocate(qbot(n_svats))
allocate(qmodf(n_svats))
allocate(reva(n_svats,2))
allocate(sc1(n_svats))
allocate(select_spu(370)); select_spu = .FALSE.
allocate(h(n_svats,nbox))                     ! vertical profile of pressure heads
allocate(submerged(n_svats,nbox))                 ! vertical profile of moisture content
allocate(theta(n_svats,nbox))                 ! moisture content

! driving forces
external_input = .TRUE.
if (external_input) then
   open(unit = 28, file = 'D:/leander/GWSobek/GWSobek/branches/sss_prototype/src/voorMariusOrig/input.dat', status = 'old')
   read(28,*) ! header
   do i = 1, ntime
      read(28,*) times(i), qrain(i), qevap(i), qrot(i), gwls(i)
   end do
   close(unit = 28)
else

   qrain(1:60)      = 0.45d0     ! cm/d
   qrain(61:80)     = 0.26d0
   qrain(81:100)    = 0.32d0

   ! basis
   qrain(101:ntime) = 0.d0
   qevap(:)         = 1.0d0

   ! alternatief
   !qrain(101:ntime) = 0.1d0
   !qevap(1:100)     = 1.0d0
   !qevap(101:ntime) = 0.0d0

   qrot             = 0.0_hp
end if

! Fill convenient struct with parameters for this sequential steady state instance
param(:)%area         = 100.d0      ! area
param(:)%top          =   0.d0      ! elevation
param(:)%init_phead   =  -1.5136d0  ! initial phead
param(:)%init_gwl     =  -3.d0      ! initial gwl
param(:)%dprz         =   1.d0      ! rootzone depth, thickness
param(:)%dtgw         =   1.d0      ! groundwater timestep
param(:)%zmax_ponding =   0.02d0    ! ponding reservoir depth
param(:)%maxinf       =   0.0032d0  ! infiltration rate limit
param(:)%soil_resist  =   1.d0      ! soil resistance
param(1)%spu          =  79         ! soil physical unit number
param(2)%spu          =  80         ! soil physical unit number

dtgw = param(1)%dtgw                           ! timestep size
dbpath = "D:/leander/MetaSWAP/test_compare_megaswap/van_hendrik_new/metaswap_a/unsa_nc2"

! set mask which db spu-files to read
do k = 1, n_svats
   select_spu(param(k)%spu) = .TRUE.
end do

! Read set of databases. Each database corresponds with a soil physical unit and a rootzone depth
!  dbpath contains all the netcdf files, dbset is the instance holding the resulting set
suc6 = dbset%readNCset(dbpath, select_spu)

! set initial h
h(1,:) = param(1)%init_phead               !l initialize the heads profile with a constant value
h(2,:) = param(2)%init_phead               ! initialize the heads profile with a constant value
theta = 0.0d0

k1 = 1
k2 = 1
allocate(results(k1:k2))

if (k1 == k2) then
   iun1 = 123
   open(unit = iun1, file = 'fort_1.csv', status = 'unknown')
   iun2 = 124
   open(unit = iun2, file = 'fort_2.csv', status = 'unknown')
else
   iun1 = 123
   open(unit = iun1, file = 'fort_1_samen.csv', status = 'unknown')
   iun2 = 124
   open(unit = iun2, file = 'fort_2_samen.csv', status = 'unknown')
endif

do k = k1, k2
   suc6 = svat(k)%initialize(param(k),dbset)   ! init svat sequential steady state instance
   call results(k)%init(nbox,nnod)
   results(k)%h = param(k)%init_phead          ! initialize the heads profile with a constant value
   results(k)%th = 0.0d0
   results(k)%gwl = param(k)%init_gwl 
   results(k)%dbpath = svat(k)%unsa%unsa_db%dbpath
end do

spu  = svat(1)%unsa%unsa_db%spu
irz  = svat(1)%unsa%unsa_db%irz
write(unsa_dbname,'(a,i3.3,a,i3.3,a)') 'unsa_',spu,'_',irz,'.nc'
call dump%init('results.nc',nbox,param(1),trim(results(1)%dbpath))
do itime = 1, ntime                   ! time-loop

   ! prepare
   do k = k1, k2
      call svat(k)%prepare(dtgw,qrain(itime),qevap(itime),qrot(itime))
   end do
   
   results%gwl=gwls(itime)
   ! dyanmic: iterations
   do iter=1, niter                   ! iteration-loop
      do k = k1, k2
         if (external_input) then
            gwl = gwls(itime)
         else
            gwl = -150.d0 ! - (itime - 1)*0.01d0                    ! heads from mf6: m+NAP; as of 02-06-2024: in cm
         end if

          call svat(k)%calc(gwl,h(k,:),qbot(k),sc1(k))
          ! storage coefficients to mf6 
          ! recharge flux to mf6
          ! mf6_solve

      end do ! k
   end do ! iter
   
   ! finalize
   do k = k1, k2
      results(k)%qbot = qbot(k)
      call svat(k)%finalize(gwl,pond(k),qrunoff(k),qmodf(k),theta(k,:),reva(k,1:2),submerged(k,:))
      results(k)%nbox      = svat(k)%unsa%maxbox
      results(k)%gwl       = svat(k)%gwl
      results(k)%pond      = svat(k)%ponding%stage
      results(k)%qrun      = svat(k)%qrun
      results(k)%qmodf     = svat(k)%qmodf
      results(k)%thbox     = theta(k,:)
      results(k)%reva      = svat(k)%evsoil
      results(k)%submerged = submerged(k,:)
      results(k)%qrain     = svat(k)%rr
      results(k)%peva      = svat(k)%ev
      results(k)%gamma     = svat(k)%unsa%gam
      results(k)%phi(:)    = svat(k)%unsa%phi(:)
      results(k)%h(:)      = svat(k)%unsa%phead(:)
      results(k)%qrch      = svat(k)%qrch
      results(k)%qrot      = svat(k)%qrot
      
      if (k == 1) write(iun1,'(i8,14(",",F15.8))') itime, gwl, h(k,1), h(k,2), theta(k,1), theta(k,2), qbot(k), sc1(k), qrot(itime), pond(k), qrunoff(k), qmodf(k), reva(k,1), reva(k,2), qrain(itime)
      if (k == 1) write(1123,'(i8,40(",",F15.8))') itime, gwl, h(k,1:18), theta(k,1:18)
      if (k == 2) write(iun2,'(i8,14(",",F15.8))') itime, gwl, h(k,1), h(k,2), theta(k,1), theta(k,2), qbot(k), sc1(k), qrot(itime), pond(k), qrunoff(k), qmodf(k), reva(k,1), reva(k,2), qrain(itime)
      
   end do
   
   write(0,'(a,i4.4,a)') 'Timestep # ', itime, ' completed  ... '
   call dump%dump(itime*1.d0, results(1))

enddo ! itime
call dump%close() 

end program ExampleSSS_coupled
