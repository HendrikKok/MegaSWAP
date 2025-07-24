module storage_formulation
use protoglobal
use database
implicit none

    real(kind=hp) :: sc1_min  = 0.001
    integer       :: iterur1  = 3            ! lower bound for smoothing sc1
    integer       :: iterur2  = 5            ! uper bound for smoothing sc1
    real(kind=hp) :: treshold = 0.00025      ! treshold level for change in head in m

    type :: t_storageFormulation
        real(kind=hp)              :: dtgw = 1.             ! groundwater time step
        real(kind=hp)              :: s, s0                 ! new and old total storage   
        real(kind=hp)              :: gwl, gwl0             ! new and old groundwater level (own groundwater level)
        real(kind=hp)              :: gwl_mf6, gwl0_mf6     ! new and old groundwater level modflow6 (incoming groundwater level)
        real(kind=hp)              :: gam_mf6               ! groundwater level index modflow6
        type(t_database), pointer  :: unsa_db => null()     ! database pointer
        real(kind=hp)              :: qmodf                 ! contribution of MODFLOW 6 to shared water balance
        real(kind=hp)              :: vcor                  ! correction for non convergence
        real(kind=hp)              :: sc1                   ! sy for MODFLOW 6
    contains
        procedure, pass :: initialize => t_storageFormulation_init       ! init storage formulation instance
    end type t_storageFormulation

contains

    function relaxation_factor(iter) result (omega)
        real(kind=hp)             :: omega
        integer, intent(in)       :: iter
        omega = min(1.d0,                 &
                  max(0.d0,               &
                    (dble(iterur2 - iter) &
                        /(iterur2 - iterur1))))
    end function relaxation_factor

    function t_storageFormulation_init(sfu, dbptr, s, s0, initial_gwl, dtgw) result (success)
        logical                                     :: success
        class(t_storageFormulation), intent(inout)  :: sfu
        type(t_database), pointer  :: dbptr   
        real(kind=hp), intent(in)  :: s, s0
        real(kind=hp), intent(in)  :: initial_gwl
        real(kind=hp), intent(in)  :: dtgw
        success = .False. 
        sfu%unsa_db => dbptr
        sfu%gwl = initial_gwl
        sfu%gwl0 = initial_gwl
        sfu%gwl_mf6 = initial_gwl
        sfu%gwl0_mf6 = initial_gwl
        sfu%s = s
        sfu%s0 = s0
        sfu%gam_mf6 = 0.d0
        sfu%vcor = 0.d0
        sfu%qmodf = 0.d0
        success = .True. 
    end function t_storageFormulation_init

end module storage_formulation