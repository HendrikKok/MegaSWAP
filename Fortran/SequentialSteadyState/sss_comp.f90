module sss_comp
use database_io
use globals
use sss
implicit none

   integer :: iter_sss
   integer :: svat_nr_max = 0
   type(t_SequentialSteadyState), allocatable :: svats(:)
   real(kind=hp), allocatable, target    :: rr_array(:)     ! rainfall forcing
   real(kind=hp), allocatable, target    :: ev_array(:)     ! evap forcing
   real(kind=hp), allocatable, target    :: evsoil_array(:) ! actual soil evap
   real(kind=hp), allocatable, target    :: evpond_array(:) ! actual ponding evap
   real(kind=hp), allocatable, target    :: qrun_array(:)   ! runoff
   real(kind=hp), allocatable, target    :: gwl_array(:)
   real(kind=hp), allocatable, target    :: phead_array(:,:)
   real(kind=hp), allocatable, target    :: vsim_array(:)
   real(kind=hp), allocatable, target    :: sc1_array(:)
   real(kind=hp), allocatable, target    :: qmodf_array(:)      
   integer,       allocatable, target    :: nbox_array(:)   ! number of non-submerged boxes           
   real(kind=hp),              target    :: dtgw

   type(t_databaseSet), target :: dbset
   character(len=*), parameter :: sversion = '0.0.0'
   integer                     :: lun_meteo = 0       ! meteo file handle (test purpose, metaswap style mete_svat.inp) 
   real(kind=hp)               :: now = 0.d0          ! days since start of the simulation, referred to when reading meteo
   integer                     :: itime = 0           ! timestep number

    type t_meteo
        real(kind=hp) :: time
        real(kind=hp) :: rr
        real(kind=hp) :: ev
    end type t_meteo
    type(t_meteo), target  :: meteo_1, meteo_2
    type(t_meteo), pointer :: meteo_new, meteo_old

contains

      subroutine sss_meteo_update()
      character(len=200) :: line_meteo
          type(t_meteo), pointer :: ptr
          integer :: yr
          do while (now >= meteo_new%time)
              ptr => meteo_old
              meteo_old => meteo_new
              meteo_new => ptr
              read(lun_meteo,'(a200)') line_meteo
              read(line_meteo,*) meteo_new%time, yr, meteo_new%rr, meteo_new%ev
          enddo
      end subroutine sss_meteo_update

      subroutine sss_initComponent()
      integer :: lun, ios, svat_nr
      character(len=200) :: line
      open(file='area_svat.inp', status='OLD', newunit=lun, iostat=ios)
      if (ios==0) then
          svat_nr_max = 0
          do while(.True.)
              read(lun,'(a200)', iostat=ios)  line
              if (ios.ne.0) exit
              if (line(1:1)=='#') cycle
              read(line(1:10),*) svat_nr
              svat_nr_max = max(svat_nr_max,svat_nr)
          enddo
          allocate(rr_array(svat_nr_max))
          allocate(ev_array(svat_nr_max))
          allocate(evsoil_array(svat_nr_max))
          allocate(evpond_array(svat_nr_max))
          allocate(gwl_array(svat_nr_max))
          allocate(vsim_array(svat_nr_max))
          allocate(qrun_array(svat_nr_max))
          allocate(sc1_array(svat_nr_max))
          allocate(nbox_array(svat_nr_max))
          allocate(qmodf_array(svat_nr_max))
      else
          ! something went wrong opening the area_svat.inp file
          return
      endif
      close(lun)
      end subroutine sss_initComponent

      subroutine sss_initSimulation()
      type(t_sssparam) :: parameters
      integer :: lun, ios, svat_nr, yr, nbox
      character(len=200) :: line
      iter_sss = 0

      ! tbd, read parameters from a parasim.inp file
      parameters%dtgw = 1.d0           ! groundwater timestep
      parameters%dprz = 1.d0           ! rootzone thickness
      parameters%init_gwl = -3.d0      ! initial groundwater level
      parameters%init_phead = -1.513561! initial phead

      ! read the database
      if (.not.dbset%readNCset("database")) then
          return        ! reading the database went wrong
      endif

      ! read area_svat.inp
      open(file='area_svat.inp', status='OLD', newunit=lun, iostat=ios)
      write(123,'(a8,a8,2a8,a15)') 'itime', 'now', 'k', 'b', 'phead' 

      ! prepare svat data structure
      if (ios==0) then
          if (allocated(svats)) deallocate(svats)
          allocate(svats(svat_nr_max))
          nbox = ubound(dbset%dbs(1,1)%ptr%hbotb,dim=1)
          if(.not.realloc(phead_array,1,nbox,1,svat_nr_max)) then
              return ! add exception handling here
          endif

          ios=0
          do while(.True.)
              read(lun,'(a200)', iostat=ios)  line
              if (ios.ne.0) exit
              if (line(1:1)=='#') cycle
              read(line(1:10),*) svat_nr
              read(line(11:20),*) parameters%area            ! area
              read(line(21:28),*) parameters%top             ! elevation
              read(line(37:42),*) parameters%spu             ! soil physical unit number
              read(line(99:110),*) parameters%init_phead     ! initial phead
              read(line(111:122),*) parameters%init_gwl      ! initial gwl
              read(line(123:134),*) parameters%dprz          ! rootzone depth
              read(line(135:142),*) parameters%zmax_ponding  ! ponding reservoir depth
              read(line(143:150),*) parameters%maxinf        ! infiltration rate limit
              read(line(151:164),*) parameters%soil_resist   ! soil resistance

              svats(svat_nr)%rr => rr_array(svat_nr)
              svats(svat_nr)%ev => ev_array(svat_nr)
              svats(svat_nr)%evsoil => evsoil_array(svat_nr)
              svats(svat_nr)%evpond => evpond_array(svat_nr)
              svats(svat_nr)%gwl => gwl_array(svat_nr)
              svats(svat_nr)%vsim => vsim_array(svat_nr)
              svats(svat_nr)%qrun => qrun_array(svat_nr)
              svats(svat_nr)%sc1 => sc1_array(svat_nr)
              svats(svat_nr)%qmodf => qmodf_array(svat_nr)
              if (.not.svats(svat_nr)%initialize(parameters,dbset,phead_array(:,svat_nr))) then
                  !report something went wrong initializing 
                  !the sequential steady state instance
                  return
              endif
          enddo
          close(lun)
          open(file='mete_svat.inp', status='OLD', newunit=lun_meteo, iostat=ios)
          if (ios/=0) then
             write(0,*) 'Problem opening mete_svat.inp for mete input'
             return
          endif 
          read(lun_meteo,'(a200)') line
          read(line,*) meteo_1%time, yr, meteo_1%rr, meteo_1%ev
          read(lun_meteo,'(a200)') line
          read(line,*) meteo_2%time, yr, meteo_2%rr, meteo_2%ev
          meteo_old => meteo_1
          meteo_new => meteo_2
      endif
      end subroutine sss_initSimulation

      subroutine sss_saveFluxes()
      integer :: k
      do k = 1, size(svats)
          call svats(k)%save_fluxes()
      enddo
      end subroutine sss_saveFluxes

      subroutine sss_restoreFluxes()
      integer :: k
      do k = 1, size(svats)
          call svats(k)%restore_fluxes()
      enddo
      end subroutine sss_restoreFluxes

      subroutine sss_initTimestep(dt)
      real(kind=hp), intent(in) :: dt
      integer :: k, iy
      real(kind=hp) :: rr, ev, doy, wt
      call sss_meteo_update()    ! update meteo from file mete_svat.inp
      wt = (now-meteo_old%time)/(meteo_new%time-meteo_old%time)
      rr_array(:) = (1.d0 - wt) * meteo_old%rr + wt * meteo_new%rr
      ev_array(:) = (1.d0 - wt) * meteo_old%ev + wt * meteo_new%ev
      call sss_saveFluxes()
      dtgw = dt
      do k = 1, size(svats)
          svats(k)%dtgw = dt
          call svats(k)%do_unsa()
          nbox_array(k) = svats(k)%unsa%maxbox
      enddo
      end subroutine sss_initTimestep

      subroutine sss_finishTimestep()
      integer :: k
      integer :: b
      do k = 1, size(svats)
          if (.not.svats(k)%finalize_tstep()) then
              return 
          endif 
      enddo
      now = now + dtgw
      itime = itime + 1

      ! write
      do k = 1, size(svats)
         do b = 1, svats(k)%unsa%maxbox
            write(123,'(i8,f8.3,2i8,e15.5)') itime, now, k, b, svats(k)%unsa%phead(b) 
         enddo 
      enddo
      end subroutine sss_finishTimestep

      subroutine sss_solve()
      integer :: k
      do k = 1, size(svats)
          call svats(k)%do_unsa()
          nbox_array(k) = svats(k)%unsa%maxbox
      enddo
      end subroutine sss_solve

end module sss_comp
