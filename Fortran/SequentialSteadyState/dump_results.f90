module dumpres
use globals
use netcdf
use sss
implicit none

type :: t_results
   integer :: nbox, nnod
   real(kind=hp), allocatable :: h(:), th(:), thbox(:), phi(:) 
   logical, allocatable       :: submerged(:)
   character(len=:), allocatable :: dbpath
   real(kind=hp)              :: sc1, qbot, qrot, qrch, qrun, gwl, pond, qmodf, qrain, peva, reva(2), gamma
contains
   procedure, pass :: init    => t_results_init
end type t_results

type :: t_dumpnc
   integer :: ncid
   integer :: ierr = 0
   integer :: count = 0
   integer :: varid_time, varid_h, varid_th, varid_sc1
   integer :: varid_qmodf, varid_qrun, varid_gwl, varid_nbox, varid_pond 
   integer :: varid_reva_ponding, varid_reva, varid_qrain, varid_peva
   integer :: varid_qbot, varid_qrot, varid_qrch
   integer :: varid_gamma, varid_phi
   integer :: dimid_time, dimid_box
contains
   procedure, pass :: init        => t_dumpnc_init
   procedure, pass :: dump        => t_dumpnc_dump
   procedure, pass :: close       => t_dumpnc_close
end type t_dumpnc

contains

subroutine t_results_init(self, nbox, nnod)
   class(t_results), target, intent(inout) :: self
   integer, intent(in) :: nbox  ! number of boxes
   integer, intent(in) :: nnod  ! number of nodes
   self%nbox = nbox
   if (allocated(self%h)) deallocate(self%h)
   allocate(self%h(nbox))
   if (allocated(self%th)) deallocate(self%th)
   allocate(self%th(nnod))
   if (allocated(self%thbox)) deallocate(self%thbox)
   allocate(self%thbox(nbox))
   if (allocated(self%phi)) deallocate(self%phi)
   allocate(self%phi(nbox))
   if (allocated(self%submerged)) deallocate(self%submerged)
   allocate(self%submerged(nbox))
end subroutine t_results_init

subroutine t_dumpnc_init(self, fnout, nbox, param, unsa_db_str)
   class(t_dumpnc), target, intent(inout) :: self
   type(t_sssparam),        intent(in)    :: param
   character(len=*), intent(in) :: unsa_db_str
   character(len=*), intent(in) :: fnout
   integer, intent(in) :: nbox  ! number of boxes
   integer :: varid_settings    ! netcdf variable with attributes storing svat settings
   self%ierr = nf90_create(fnout, NF90_WRITE, self%ncid)
   self%ierr = nf90_def_dim(self%ncid, 'ib', nbox, self%dimid_box)         ! box dimension
   self%ierr = nf90_def_dim(self%ncid, 'time', NF90_UNLIMITED, self%dimid_time) ! time dimension

   ! Define variables
   self%ierr = nf90_def_var(self%ncid, 'phead', NF90_REAL8,(/self%dimid_box,self%dimid_time/), self%varid_h)
   self%ierr = nf90_def_var(self%ncid, 'theta', NF90_REAL8,(/self%dimid_box,self%dimid_time/), self%varid_th)
   self%ierr = nf90_def_var(self%ncid, 'sc1', NF90_REAL8,(/self%dimid_time/), self%varid_sc1)
   self%ierr = nf90_def_var(self%ncid, 'qbot', NF90_REAL8,(/self%dimid_time/), self%varid_qbot)
   self%ierr = nf90_def_var(self%ncid, 'qrot', NF90_REAL8,(/self%dimid_time/), self%varid_qrot)
   self%ierr = nf90_def_var(self%ncid, 'qrch', NF90_REAL8,(/self%dimid_time/), self%varid_qrch)
   self%ierr = nf90_def_var(self%ncid, 'qrun', NF90_REAL8,(/self%dimid_time/), self%varid_qrun)
   self%ierr = nf90_def_var(self%ncid, 'gwl', NF90_REAL8,(/self%dimid_time/), self%varid_gwl)
   self%ierr = nf90_def_var(self%ncid, 'nbox', NF90_INT4,(/self%dimid_time/), self%varid_nbox)
   self%ierr = nf90_def_var(self%ncid, 'pond', NF90_REAL8,(/self%dimid_time/), self%varid_pond)
   self%ierr = nf90_def_var(self%ncid, 'qmodf', NF90_REAL8,(/self%dimid_time/), self%varid_qmodf)
   self%ierr = nf90_def_var(self%ncid, 'qrain', NF90_REAL8,(/self%dimid_time/), self%varid_qrain)
   self%ierr = nf90_def_var(self%ncid, 'peva', NF90_REAL8,(/self%dimid_time/), self%varid_peva)
   self%ierr = nf90_def_var(self%ncid, 'reva', NF90_REAL8,(/self%dimid_time/), self%varid_reva)
   self%ierr = nf90_def_var(self%ncid, 'ponding_reva', NF90_REAL8,(/self%dimid_time/), self%varid_reva_ponding)
   self%ierr = nf90_def_var(self%ncid, 'gamma' , NF90_REAL8,(/self%dimid_time/), self%varid_gamma)
   self%ierr = nf90_def_var(self%ncid, 'phi', NF90_REAL8,(/self%dimid_box,self%dimid_time/), self%varid_phi)
   self%ierr = nf90_def_var(self%ncid, 'time', NF90_REAL8,(/self%dimid_time/), self%varid_time)

   ! create a variable for settings, holding no data, only attributes
   self%ierr = nf90_def_var(self%ncid, 'settings', NF90_INT, varid=varid_settings)
   self%ierr = nf90_put_att(self%ncid, varid_settings, 'unsa_db', trim(unsa_db_str))
   self%ierr = nf90_put_att(self%ncid, varid_settings, 'spu', param%spu)
   self%ierr = nf90_put_att(self%ncid, varid_settings, 'dtgw', param%dtgw)
   self%ierr = nf90_put_att(self%ncid, varid_settings, 'dprz', param%dprz)
   self%ierr = nf90_put_att(self%ncid, varid_settings, 'area', param%area)
   self%ierr = nf90_put_att(self%ncid, varid_settings, 'top', param%top)
   self%ierr = nf90_put_att(self%ncid, varid_settings, 'init_gwl', param%init_gwl)
   self%ierr = nf90_put_att(self%ncid, varid_settings, 'init_phead', param%init_phead)
   self%ierr = nf90_put_att(self%ncid, varid_settings, 'zmax_ponding', param%zmax_ponding)
   self%ierr = nf90_put_att(self%ncid, varid_settings, 'soil_resist', param%soil_resist)
   self%ierr = nf90_put_att(self%ncid, varid_settings, 'maxinf', param%maxinf)

   self%ierr = nf90_enddef(self%ncid)
end subroutine t_dumpnc_init

subroutine t_dumpnc_dump(self, time, results)
   class(t_dumpnc), target, intent(inout) :: self
   real(kind=hp),           intent(in)    :: time
   type(t_results),         intent(in)    :: results
   integer :: nbox
   self%count = self%count + 1
   nbox=size(results%h)
   self%ierr = nf90_put_var(self%ncid, self%varid_h, results%h(:), start=(/1,self%count/), count=(/nbox,1/)) 
   self%ierr = nf90_put_var(self%ncid, self%varid_th, results%thbox(:), start=(/1,self%count/), count=(/nbox,1/)) 
   self%ierr = nf90_put_var(self%ncid, self%varid_sc1, results%sc1,start=(/self%count/)) 
   self%ierr = nf90_put_var(self%ncid, self%varid_qbot, results%qbot,start=(/self%count/)) 
   self%ierr = nf90_put_var(self%ncid, self%varid_qrot, results%qrot,start=(/self%count/)) 
   self%ierr = nf90_put_var(self%ncid, self%varid_qrch, results%qrch,start=(/self%count/)) 
   self%ierr = nf90_put_var(self%ncid, self%varid_qrun, results%qrun,start=(/self%count/)) 
   self%ierr = nf90_put_var(self%ncid, self%varid_gwl, results%gwl,start=(/self%count/)) 
   self%ierr = nf90_put_var(self%ncid, self%varid_nbox, results%nbox,start=(/self%count/)) 
   self%ierr = nf90_put_var(self%ncid, self%varid_pond, results%pond,start=(/self%count/)) 
   self%ierr = nf90_put_var(self%ncid, self%varid_qmodf, results%qmodf,start=(/self%count/)) 
   self%ierr = nf90_put_var(self%ncid, self%varid_qrain, results%qrain,start=(/self%count/))
   self%ierr = nf90_put_var(self%ncid, self%varid_peva, results%peva,start=(/self%count/))
   self%ierr = nf90_put_var(self%ncid, self%varid_reva, results%reva(1),start=(/self%count/))
   self%ierr = nf90_put_var(self%ncid, self%varid_reva_ponding, results%reva(2),start=(/self%count/))
   self%ierr = nf90_put_var(self%ncid, self%varid_gamma, results%gamma,start=(/self%count/))
   self%ierr = nf90_put_var(self%ncid, self%varid_phi, results%phi(:), start=(/1,self%count/), count=(/nbox,1/)) 
   self%ierr = nf90_put_var(self%ncid, self%varid_time, time,start=(/self%count/)) 
end subroutine t_dumpnc_dump

subroutine t_dumpnc_close(self)
   class(t_dumpnc), target, intent(inout) :: self
   self%ierr = nf90_close(self%ncid)
   self%ncid = 0
end subroutine t_dumpnc_close

end module dumpres
