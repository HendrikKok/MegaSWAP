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
use dumpres
use database_io
implicit none

type (t_SequentialSteadyStateSingle), dimension(:), allocatable :: svat
type (t_sssparam),                    dimension(:), allocatable :: param
type(t_databaseSet),                                target      :: dbset

! results to netcdf file for check
type (t_dumpnc) :: dump
type (t_results) :: results

character(len=:), allocatable :: dbpath
real(kind=hp), dimension(:,:), allocatable :: theta, h
real(kind=hp), dimension(:), allocatable :: pond, qrunoff, qbot, qmodf, reva, sc1
real(kind=hp) :: gwl, dtgw, qrot
logical :: suc6
integer :: n_svats, itime, iter, k, k1, k2, iun1, iun2
integer, parameter :: ntime = 200
integer, parameter :: niter = 1
integer, parameter :: nbox = 18
real(kind=hp), dimension(ntime) :: qrain, qevap

qrain(1:60)      = 0.45d0     ! cm/d
qrain(61:80)     = 0.26d0
qrain(81:100)    = 0.32d0

! basis
qrain(101:ntime) = 0.d0
qevap(:)         = 0.4d0

! alternatief
!qrain(101:ntime) = 0.1d0
!qevap(1:100)     = 1.0d0
!qevap(101:ntime) = 0.0d0


qrot             = 0.0_hp

dtgw = 1.d0                           ! timestep size

! Read set of databases. Each database corresponds with a soil physical unit and a rootzone depth
! dbpath contains all the netcdf files, dbset is the instance holding the resulting set
!!!dbpath = "D:/leander/MetaSWAP/test_compare_megaswap/van_hendrik_new/metaswap_a/unsa_nc"
dbpath = "D:/leander/MetaSWAP/test_compare_megaswap/van_hendrik_new/metaswap_a/unsa_nc"
suc6 = dbset%readNCset(dbpath)

n_svats = 2
allocate(param(n_svats))
allocate(svat(n_svats))
allocate(pond(n_svats))
allocate(qrunoff(n_svats))
allocate(qbot(n_svats))
allocate(qmodf(n_svats))
allocate(reva(n_svats))
allocate(sc1(n_svats))

! Fill convenient struct with parameters for this sequential steady state instance
param(:)%area         = 100.d0      ! area
param(:)%top          =   0.d0      ! elevation
param(:)%init_phead   =  -1.5136d0  ! initial phead
param(:)%init_gwl     =  -3.d0      ! initial gwl
param(:)%dprz         =   1.d0      ! rootzone depth, thickness
param(:)%zmax_ponding =   0.02d0    ! ponding reservoir depth
param(:)%maxinf       =   0.0032d0  ! infiltration rate limit
param(:)%soil_resist  =   1.d0      ! soil resistance
param(1)%spu          =  79         ! soil physical unit number
param(2)%spu          =  80         ! soil physical unit number

!suc6 = svat%initialize(param,dbset)   ! init svat sequential steady state instance

allocate(h(n_svats,nbox))                     ! vertical profile of pressure heads
allocate(theta(n_svats,nbox))                 ! vertical profile of moisture content

! set timestep
h(1,:) = param(1)%init_phead               ! initialize the heads profile with a constant value
h(2,:) = param(2)%init_phead               ! initialize the heads profile with a constant value
theta = 0.0d0

k1 = 1
k2 = 2

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

! open some files for output
call results%init(nbox)
call dump%init('results.nc') 

do k = k1, k2
   suc6 = svat(k)%initialize(param(k),dbset)   ! init svat sequential steady state instance
end do

do itime = 1, ntime                   ! time-loop
!  results%gwl = -1.d0 ! - (itime - 1)*0.01d0          ! heads from mf6: m+NAP
   results%gwl = -200.d0 + (itime - 1)*1.d0           ! heads from mf6: m+NAP

   ! prepare
   do k = k1, k2
      call svat(k)%prepare(dtgw,qrain(itime),qevap(itime))
   end do
   
   ! dyanmic: iterations
   do iter=1, niter                   ! iteration-loop
      do k = k1, k2
         !call svat(k)%calc(gwl,h(k,:),theta(k,:),qbot(k),sc1(k))
          call svat(k)%calc(results%gwl, &
                         results%h,      &
                         results%th,     &
                         results%qbot,   &
                         results%sc1)   ! set initial groundwater level as the current level, no modflow interaction

          ! storage coefficients to mf6
          ! recharge flux to mf6
          ! mf6_solve
      end do ! k
   end do ! iter
   
   ! finalize
   do k = k1, k2
!     call svat(k)%finalize(qrot,gwl,pond(k),qrunoff(k),qmodf(k),reva(k))
      call svat(k)%finalize(qrot,       &
                         results%gwl,   &
                         results%pond,  &
                         results%qrun,  &
                         results%qmodf, &
                         results%reva)
      results%qrain=qrain(itime)
      results%peva=qevap(itime)
      
!      if (k == 1) write(iun1,'(i8,13(",",F15.8))') itime, gwl, h(k,1), h(k,2), theta(k,1), theta(k,2), qbot(k), sc1(k), qrot, pond(k), qrunoff(k), qmodf(k), reva(k), qrain(itime)
!      if (k == 1) write(1123,'(i8,40(",",F15.8))') itime, gwl, h(k,1:18), theta(k,1:18)
!      if (k == 2) write(iun2,'(i8,13(",",F15.8))') itime, gwl, h(k,1), h(k,2), theta(k,1), theta(k,2), qbot(k), sc1(k), qrot, pond(k), qrunoff(k), qmodf(k), reva(k), qrain(itime)
      results%h(:) = svat(k)%unsa%phead(:) ! rl666
      
   end do
   
   write(0,'(a,i4.4,a)') 'Timestep # ', itime, ' completed  ... '
   call dump%dump(itime*1.d0, results)

enddo ! itime
call dump%close() 

end program ExampleSSS_coupled
